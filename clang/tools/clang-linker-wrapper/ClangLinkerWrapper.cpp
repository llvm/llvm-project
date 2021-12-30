//===-- clang-linker-wrapper/ClangLinkerWrapper.cpp - wrapper over linker-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -help)
// will be hidden.
static cl::OptionCategory
    ClangLinkerWrapperCategory("clang-linker-wrapper options");

static cl::opt<std::string> LinkerUserPath("linker-path",
                                           cl::desc("Path of linker binary"),
                                           cl::cat(ClangLinkerWrapperCategory));

// Do not parse linker options
static cl::list<std::string>
    LinkerArgs(cl::Sink, cl::desc("<options to be passed to linker>..."));

static Error runLinker(std::string LinkerPath,
                       SmallVectorImpl<std::string> &Args) {
  std::vector<StringRef> LinkerArgs;
  LinkerArgs.push_back(LinkerPath);
  for (auto &Arg : Args)
    LinkerArgs.push_back(Arg);

  if (sys::ExecuteAndWait(LinkerPath, LinkerArgs))
    return createStringError(inconvertibleErrorCode(), "'linker' failed");
  return Error::success();
}

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-linker-wrapper") << '\n';
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::SetVersionPrinter(PrintVersion);
  cl::HideUnrelatedOptions(ClangLinkerWrapperCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A wrapper utility over the host linker. It scans the input files for\n"
      "sections that require additional processing prior to linking. The tool\n"
      "will then transparently pass all arguments and input to the specified\n"
      "host linker to create the final binary.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    exit(EXIT_FAILURE);
  };

  // TODO: Scan input object files for offloading sections and extract them.
  // TODO: Perform appropriate device linking action.
  // TODO: Wrap device image in a host binary and pass it to the linker.
  WithColor::warning(errs(), argv[0]) << "Offload linking not yet supported.\n";

  SmallVector<std::string, 0> Argv;
  for (const std::string &Arg : LinkerArgs)
    Argv.push_back(Arg);

  if (Error Err = runLinker(LinkerUserPath, Argv))
    reportError(std::move(Err));

  return EXIT_SUCCESS;
}
