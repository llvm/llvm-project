//===-- fc1_main.cpp - Flang FC1 Compiler Frontend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the flang -fc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticBuffer.h"
#include "flang/FrontendTool/Utils.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdio>

using namespace Fortran::frontend;

int fc1_main(llvm::ArrayRef<const char *> argv, const char *argv0) {
  // Create CompilerInstance
  std::unique_ptr<CompilerInstance> flang(new CompilerInstance());

  // Create DiagnosticsEngine for the frontend driver
  flang->createDiagnostics();
  if (!flang->hasDiagnostics())
    return 1;

  // We will buffer diagnostics from argument parsing so that we can output
  // them using a well formed diagnostic object.
  TextDiagnosticBuffer *diagsBuffer = new TextDiagnosticBuffer;

  // Create CompilerInvocation - use a dedicated instance of DiagnosticsEngine
  // for parsing the arguments
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      new clang::DiagnosticOptions();
  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagsBuffer);
  bool success = CompilerInvocation::createFromArgs(flang->getInvocation(),
                                                    argv, diags, argv0);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  diagsBuffer->flushDiagnostics(flang->getDiagnostics());

  if (!success)
    return 1;

  // Execute the frontend actions.
  success = executeCompilerInvocation(flang.get());

  // Delete output files to free Compiler Instance
  flang->clearOutputFiles(/*EraseFiles=*/false);

  return !success;
}
