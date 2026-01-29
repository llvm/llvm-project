//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Driver for SPIR-V linking
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "Config.h"
#include "LTO.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Version.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace lld::spirv {

Config config;

static void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
}

static void printVersion(llvm::raw_ostream &os) {
  os << getLLDVersion() << "\n";
}

static void printHelp() {
  llvm::outs() << "USAGE: spirv-lld [options] file...\n\n"
               << "OPTIONS:\n"
               << "  -o <path>     Output file path\n"
               << "  --version     Print version information\n"
               << "  --help        Print this help\n";
}

bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  auto *context = new CommonLinkerContext;
  context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  context->e.cleanupCallback = []() { config = Config(); };

  initLLVM();

  // Handle --version and --help
  for (size_t i = 1; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == "--version") {
      printVersion(stdoutOS);
      return true;
    }
    if (arg == "--help") {
      printHelp();
      return true;
    }
  }

  // Parse args: -o output, rest are inputs
  std::string outputFile = "a.out.spv";
  std::vector<MemoryBufferRef> inputs;

  for (size_t i = 1; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == "-o" && i + 1 < args.size()) {
      outputFile = args[++i];
    } else {
      ErrorOr<std::unique_ptr<MemoryBuffer>> mbOrErr =
          MemoryBuffer::getFile(arg);
      if (auto ec = mbOrErr.getError()) {
        error("cannot open " + arg + ": " + ec.message());
        continue;
      }
      // Only support LTO for now
      std::unique_ptr<MemoryBuffer> &mb = *mbOrErr;
      if (llvm::identify_magic(mb->getBuffer()) != llvm::file_magic::bitcode) {
        error(arg + ": not a bitcode file");
        continue;
      }
      inputs.push_back(mb->getMemBufferRef());
      make<std::unique_ptr<MemoryBuffer>>(std::move(mb));
    }
  }

  if (inputs.empty()) {
    error("no input files");
    return false;
  }

  BitcodeCompiler compiler(config);
  for (MemoryBufferRef mb : inputs)
    compiler.add(mb);

  std::vector<char> spirvBinary = compiler.compile();
  if (errorCount())
    return false;

  std::error_code ec;
  raw_fd_ostream os(outputFile, ec, sys::fs::OF_None);
  if (ec) {
    error("cannot open output file " + outputFile + ": " + ec.message());
    return false;
  }
  os.write(spirvBinary.data(), spirvBinary.size());

  return errorCount() == 0;
}

} // namespace lld::spirv
