//===--- CompilerInstance.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

CompilerInstance::CompilerInstance() : invocation_(new CompilerInvocation()) {}

CompilerInstance::~CompilerInstance() = default;

void CompilerInstance::CreateDiagnostics(
    clang::DiagnosticConsumer *client, bool shouldOwnClient) {
  diagnostics_ =
      CreateDiagnostics(&GetDiagnosticOpts(), client, shouldOwnClient);
}

clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
CompilerInstance::CreateDiagnostics(clang::DiagnosticOptions *opts,
    clang::DiagnosticConsumer *client, bool shouldOwnClient) {
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      new clang::DiagnosticsEngine(diagID, opts));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (client) {
    diags->setClient(client, shouldOwnClient);
  } else {
    diags->setClient(new TextDiagnosticPrinter(llvm::errs(), opts));
  }
  return diags;
}
