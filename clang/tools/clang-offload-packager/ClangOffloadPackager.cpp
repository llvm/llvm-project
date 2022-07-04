//===-- clang-offload-packager/ClangOffloadPackager.cpp - file bundler ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool takes several device object files and bundles them into a single
// binary image using a custom binary format. This is intended to be used to
// embed many device files into an application to create a fat binary.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::OptionCategory
    ClangOffloadPackagerCategory("clang-offload-packager options");

static cl::opt<std::string> OutputFile("o", cl::Required,
                                       cl::desc("Write output to <file>."),
                                       cl::value_desc("file"),
                                       cl::cat(ClangOffloadPackagerCategory));

static cl::list<std::string>
    DeviceImages("image",
                 cl::desc("List of key and value arguments. Required keywords "
                          "are 'file' and 'triple'."),
                 cl::value_desc("<key>=<value>,..."),
                 cl::cat(ClangOffloadPackagerCategory));

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-offload-packager") << '\n';
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(ClangOffloadPackagerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility for bundling several object files into a single binary.\n"
      "The output binary can then be embedded into the host section table\n"
      "to create a fatbinary containing offloading code.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return EXIT_FAILURE;
  };

  SmallVector<char, 1024> BinaryData;
  raw_svector_ostream OS(BinaryData);
  for (StringRef Image : DeviceImages) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);

    StringMap<StringRef> Args;
    for (StringRef Arg : llvm::split(Image, ",")) {
      auto KeyAndValue = Arg.split("=");
      if (Args.count(KeyAndValue.first))
        Args[KeyAndValue.first] =
            Saver.save(Args[KeyAndValue.first] + "," + KeyAndValue.second);
      else
        Args[KeyAndValue.first] = KeyAndValue.second;
    }

    if (!Args.count("triple") || !Args.count("file"))
      return reportError(createStringError(
          inconvertibleErrorCode(),
          "'file' and 'triple' are required image arguments"));

    OffloadBinary::OffloadingImage ImageBinary{};
    std::unique_ptr<llvm::MemoryBuffer> DeviceImage;
    for (const auto &KeyAndValue : Args) {
      StringRef Key = KeyAndValue.getKey();
      if (Key == "file") {
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ObjectOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(KeyAndValue.getValue());
        if (std::error_code EC = ObjectOrErr.getError())
          return reportError(errorCodeToError(EC));

        // Clang uses the '.o' suffix for LTO bitcode.
        if (identify_magic((*ObjectOrErr)->getBuffer()) == file_magic::bitcode)
          ImageBinary.TheImageKind = object::IMG_Bitcode;
        else
          ImageBinary.TheImageKind = getImageKind(
              sys::path::extension(KeyAndValue.getValue()).drop_front());
        ImageBinary.Image = std::move(*ObjectOrErr);
      } else if (Key == "kind") {
        ImageBinary.TheOffloadKind = getOffloadKind(KeyAndValue.getValue());
      } else {
        ImageBinary.StringData[Key] = KeyAndValue.getValue();
      }
    }
    std::unique_ptr<MemoryBuffer> Buffer = OffloadBinary::write(ImageBinary);
    if (Buffer->getBufferSize() % OffloadBinary::getAlignment() != 0)
      return reportError(
          createStringError(inconvertibleErrorCode(),
                            "Offload binary has invalid size alignment"));
    OS << Buffer->getBuffer();
  }

  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(OutputFile, BinaryData.size());
  if (!OutputOrErr)
    return reportError(OutputOrErr.takeError());
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  std::copy(BinaryData.begin(), BinaryData.end(), Output->getBufferStart());
  if (Error E = Output->commit())
    return reportError(std::move(E));
}
