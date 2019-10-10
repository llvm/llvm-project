//===-- clang-offload-wrapper/ClangOffloadWrapper.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the offload wrapper tool. It takes offload target binaries
/// as input and creates wrapper bitcode file containing target binaries
/// packaged as data.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadWrapperCategory("clang-offload-wrapper options");

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("Output filename"),
                                   cl::value_desc("filename"),
                                   cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<input files>"),
                                    cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    Target("target", cl::Required,
           cl::desc("Target triple for the output module"),
           cl::value_desc("triple"), cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string>
    OffloadTargets("offload-targets", cl::CommaSeparated, cl::OneOrMore,
                   cl::desc("Comma-separated list of device target triples"),
                   cl::value_desc("triples"),
                   cl::cat(ClangOffloadWrapperCategory));

namespace {

class BinaryWrapper {
public:
  // Binary descriptor. The first field is the a reference to the binary bits,
  // and the second is the target triple the binary was built for.
  using BinaryDesc = std::pair<ArrayRef<char>, StringRef>;

private:
  LLVMContext C;
  Module M;

  // Saver for generated strings.
  BumpPtrAllocator Alloc;
  UniqueStringSaver SS;

private:
  void createImages(ArrayRef<BinaryDesc> Binaries) {
    for (const BinaryDesc &Bin : Binaries) {
      StringRef SectionName = SS.save(".omp_offloading." + Bin.second);

      auto *DataC = ConstantDataArray::get(C, Bin.first);
      auto *ImageB =
          new GlobalVariable(M, DataC->getType(), /*isConstant=*/true,
                             GlobalVariable::ExternalLinkage, DataC,
                             ".omp_offloading.img_start." + Bin.second);
      ImageB->setSection(SectionName);
      ImageB->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      ImageB->setVisibility(llvm::GlobalValue::HiddenVisibility);

      auto *EmptyC =
          ConstantAggregateZero::get(ArrayType::get(Type::getInt8Ty(C), 0u));
      auto *ImageE =
          new GlobalVariable(M, EmptyC->getType(), /*isConstant=*/true,
                             GlobalVariable::ExternalLinkage, EmptyC,
                             ".omp_offloading.img_end." + Bin.second);
      ImageE->setSection(SectionName);
      ImageE->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      ImageE->setVisibility(GlobalValue::HiddenVisibility);
    }
  }

public:
  BinaryWrapper(StringRef Target) : M("offload.wrapper.object", C), SS(Alloc) {
    M.setTargetTriple(Target);
  }

  const Module &wrapBinaries(ArrayRef<BinaryDesc> Binaries) {
    createImages(Binaries);
    return M;
  }
};

} // anonymous namespace

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadWrapperCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-wrapper") << '\n';
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to create a wrapper bitcode for offload target binaries. Takes "
      "offload\ntarget binaries as input and produces bitcode file containing "
      "target binaries packaged\nas data.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
  };

  if (Triple(Target).getArch() == Triple::UnknownArch) {
    reportError(createStringError(
        errc::invalid_argument, "'" + Target + "': unsupported target triple"));
    return 1;
  }

  if (Inputs.size() != OffloadTargets.size()) {
    reportError(createStringError(
        errc::invalid_argument,
        "number of input files and offload targets should match"));
    return 1;
  }

  // Read device binaries.
  SmallVector<std::unique_ptr<MemoryBuffer>, 4u> Buffers;
  SmallVector<BinaryWrapper::BinaryDesc, 4u> Images;
  Buffers.reserve(Inputs.size());
  Images.reserve(Inputs.size());
  for (unsigned I = 0; I < Inputs.size(); ++I) {
    const std::string &File = Inputs[I];
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFileOrSTDIN(File);
    if (!BufOrErr) {
      reportError(createFileError(File, BufOrErr.getError()));
      return 1;
    }
    const std::unique_ptr<MemoryBuffer> &Buf =
        Buffers.emplace_back(std::move(*BufOrErr));
    Images.emplace_back(
        makeArrayRef(Buf->getBufferStart(), Buf->getBufferSize()),
        OffloadTargets[I]);
  }

  // Create the output file to write the resulting bitcode to.
  std::error_code EC;
  ToolOutputFile Out(Output, EC, sys::fs::OF_None);
  if (EC) {
    reportError(createFileError(Output, EC));
    return 1;
  }

  // Create a wrapper for device binaries and write its bitcode to the file.
  WriteBitcodeToFile(BinaryWrapper(Target).wrapBinaries(
                         makeArrayRef(Images.data(), Images.size())),
                     Out.os());
  if (Out.os().has_error()) {
    reportError(createFileError(Output, Out.os().error()));
    return 1;
  }

  // Success.
  Out.keep();
  return 0;
}
