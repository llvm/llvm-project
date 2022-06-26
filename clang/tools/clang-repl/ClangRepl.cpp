//===--- tools/clang-repl/ClangRepl.cpp - clang-repl - the Clang REPL -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements a REPL tool on top of clang.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h" // llvm_shutdown
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h" // llvm::Initialize*

static llvm::cl::list<std::string>
    ClangArgs("Xcc",
              llvm::cl::desc("Argument to pass to the CompilerInvocation"),
              llvm::cl::CommaSeparated);
static llvm::cl::opt<bool> OptHostSupportsJit("host-supports-jit",
                                              llvm::cl::Hidden);
static llvm::cl::list<std::string> OptInputs(llvm::cl::Positional,
                                             llvm::cl::desc("[code to run]"));

static void LLVMErrorHandler(void *UserData, const char *Message,
                             bool GenCrashDiag) {
  auto &Diags = *static_cast<clang::DiagnosticsEngine *>(UserData);

  Diags.Report(clang::diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error. Otherwise, exit with status 1.

  exit(GenCrashDiag ? 70 : 1);
}

// If we are running with -verify a reported has to be returned as unsuccess.
// This is relevant especially for the test suite.
static int checkDiagErrors(const clang::CompilerInstance *CI) {
  unsigned Errs = CI->getDiagnostics().getClient()->getNumErrors();
  if (CI->getDiagnosticOpts().VerifyDiagnostics) {
    // If there was an error that came from the verifier we must return 1 as
    // an exit code for the process. This will make the test fail as expected.
    clang::DiagnosticConsumer *Client = CI->getDiagnostics().getClient();
    Client->EndSourceFile();
    Errs = Client->getNumErrors();

    // The interpreter expects BeginSourceFile/EndSourceFiles to be balanced.
    Client->BeginSourceFile(CI->getLangOpts(), &CI->getPreprocessor());
  }
  return Errs ? EXIT_FAILURE : EXIT_SUCCESS;
}

llvm::ExitOnError ExitOnErr;
int main(int argc, const char **argv) {
  ExitOnErr.setBanner("clang-repl: ");
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  if (OptHostSupportsJit) {
    auto J = llvm::orc::LLJITBuilder().create();
    if (J)
      llvm::outs() << "true\n";
    else {
      llvm::consumeError(J.takeError());
      llvm::outs() << "false\n";
    }
    return 0;
  }

  // FIXME: Investigate if we could use runToolOnCodeWithArgs from tooling. It
  // can replace the boilerplate code for creation of the compiler instance.
  auto CI = ExitOnErr(clang::IncrementalCompilerBuilder::create(ClangArgv));

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                    static_cast<void *>(&CI->getDiagnostics()));

  // Load any requested plugins.
  CI->LoadRequestedPlugins();

  auto Interp = ExitOnErr(clang::Interpreter::create(std::move(CI)));
  for (const std::string &input : OptInputs) {
    if (auto Err = Interp->ParseAndExecute(input))
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
  }

  if (OptInputs.empty()) {
    llvm::LineEditor LE("clang-repl");
    // FIXME: Add LE.setListCompleter
    while (llvm::Optional<std::string> Line = LE.readLine()) {
      if (*Line == "quit")
        break;
      if (auto Err = Interp->ParseAndExecute(*Line))
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
    }
  }

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  return checkDiagErrors(Interp->getCompilerInstance());
}
