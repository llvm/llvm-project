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

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/FrontendTool/Utils.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"

#include <cstdio>

using namespace Fortran::frontend;

int fc1_main(llvm::ArrayRef<const char *> argv, const char *argv0) {
  // Create CompilerInstance
  std::unique_ptr<CompilerInstance> flang(new CompilerInstance());

  // Create DiagnosticsEngine for the frontend driver
  flang->CreateDiagnostics();
  if (!flang->HasDiagnostics())
    return 1;

  // Create CompilerInvocation - use a dedicated instance of DiagnosticsEngine
  // for parsing the arguments
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      new clang::DiagnosticOptions();
  clang::TextDiagnosticBuffer *diagsBuffer = new clang::TextDiagnosticBuffer;
  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagsBuffer);
  bool success =
      CompilerInvocation::CreateFromArgs(flang->GetInvocation(), argv, diags);

  diagsBuffer->FlushDiagnostics(flang->getDiagnostics());
  if (!success)
    return 1;

  // Execute the frontend actions.
  success = ExecuteCompilerInvocation(flang.get());

  return !success;
}
