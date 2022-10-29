//===-- driver.cpp - Flang Driver -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the flang driver; it is a thin wrapper
// for functionality in the Driver flang library.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"

using llvm::StringRef;

// main frontend method. Lives inside fc1_main.cpp
extern int fc1_main(llvm::ArrayRef<const char *> argv, const char *argv0);

std::string getExecutablePath(const char *argv0) {
  // This just needs to be some symbol in the binary
  void *p = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance
static clang::DiagnosticOptions *
createAndPopulateDiagOpts(llvm::ArrayRef<const char *> argv) {
  auto *diagOpts = new clang::DiagnosticOptions;

  // Ignore missingArgCount and the return value of ParseDiagnosticArgs.
  // Any errors that would be diagnosed here will also be diagnosed later,
  // when the DiagnosticsEngine actually exists.
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = clang::driver::getDriverOptTable().ParseArgs(
      argv.slice(1), missingArgIndex, missingArgCount,
      /*FlagsToInclude=*/clang::driver::options::FlangOption);

  (void)Fortran::frontend::parseDiagnosticArgs(*diagOpts, args);

  return diagOpts;
}

static int executeFC1Tool(llvm::SmallVectorImpl<const char *> &argV) {
  llvm::StringRef tool = argV[1];
  if (tool == "-fc1")
    return fc1_main(makeArrayRef(argV).slice(2), argV[0]);

  // Reject unknown tools.
  // ATM it only supports fc1. Any fc1[*] is rejected.
  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-fc1'.\n";
  return 1;
}

static void ExpandResponseFiles(
    llvm::StringSaver &saver, llvm::SmallVectorImpl<const char *> &args) {
  // We're defaulting to the GNU syntax, since we don't have a CL mode.
  llvm::cl::TokenizerCallback tokenizer = &llvm::cl::TokenizeGNUCommandLine;
  llvm::cl::ExpansionContext ExpCtx(saver.getAllocator(), tokenizer);
  ExpCtx.expandResponseFiles(args);
}

int main(int argc, const char **argv) {

  // Initialize variables to call the driver
  llvm::InitLLVM x(argc, argv);
  llvm::SmallVector<const char *, 256> args(argv, argv + argc);

  clang::driver::ParsedClangName targetandMode("flang", "--driver-mode=flang");
  std::string driverPath = getExecutablePath(args[0]);

  llvm::BumpPtrAllocator a;
  llvm::StringSaver saver(a);
  ExpandResponseFiles(saver, args);

  // Check if flang-new is in the frontend mode
  auto firstArg = std::find_if(
      args.begin() + 1, args.end(), [](const char *a) { return a != nullptr; });
  if (firstArg != args.end()) {
    if (llvm::StringRef(args[1]).startswith("-cc1")) {
      llvm::errs() << "error: unknown integrated tool '" << args[1] << "'. "
                   << "Valid tools include '-fc1'.\n";
      return 1;
    }
    // Call flang-new frontend
    if (llvm::StringRef(args[1]).startswith("-fc1")) {
      return executeFC1Tool(args);
    }
  }

  // Not in the frontend mode - continue in the compiler driver mode.

  // Create DiagnosticsEngine for the compiler driver
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      createAndPopulateDiagOpts(args);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  Fortran::frontend::TextDiagnosticPrinter *diagClient =
      new Fortran::frontend::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

  diagClient->setPrefix(
      std::string(llvm::sys::path::stem(getExecutablePath(args[0]))));

  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

  // Prepare the driver
  clang::driver::Driver theDriver(driverPath,
      llvm::sys::getDefaultTargetTriple(), diags, "flang LLVM compiler");
  theDriver.setTargetAndMode(targetandMode);
  std::unique_ptr<clang::driver::Compilation> c(
      theDriver.BuildCompilation(args));
  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
      failingCommands;

  // Run the driver
  int res = 1;
  bool isCrash = false;
  res = theDriver.ExecuteCompilation(*c, failingCommands);

  for (const auto &p : failingCommands) {
    int commandRes = p.first;
    const clang::driver::Command *failingCommand = p.second;
    if (!res)
      res = commandRes;

    // If result status is < 0 (e.g. when sys::ExecuteAndWait returns -1),
    // then the driver command signalled an error. On Windows, abort will
    // return an exit code of 3. In these cases, generate additional diagnostic
    // information if possible.
    isCrash = commandRes < 0;
#ifdef _WIN32
    isCrash |= commandRes == 3;
#endif
    if (isCrash) {
      theDriver.generateCompilationDiagnostics(*c, *failingCommand);
      break;
    }
  }

  diags.getClient()->finish();

  // If we have multiple failing commands, we return the result of the first
  // failing command.
  return res;
}
