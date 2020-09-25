//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

Sema::DeviceDiagBuilder Sema::SYCLDiagIfDeviceCode(SourceLocation Loc,
                                                   unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(getCurLexicalContext());
  DeviceDiagBuilder::Kind DiagKind = [this, FD] {
    if (!FD)
      return DeviceDiagBuilder::K_Nop;
    if (getEmissionStatus(FD) == Sema::FunctionEmissionStatus::Emitted)
      return DeviceDiagBuilder::K_ImmediateWithCallStack;
    return DeviceDiagBuilder::K_Deferred;
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID, FD, *this);
}

bool Sema::checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  assert(Callee && "Callee may not be null.");

  // Errors in unevaluated context don't need to be generated,
  // so we can safely skip them.
  if (isUnevaluatedContext() || isConstantEvaluated())
    return true;

  DeviceDiagBuilder::Kind DiagKind = DeviceDiagBuilder::K_Nop;

  return DiagKind != DeviceDiagBuilder::K_Immediate &&
         DiagKind != DeviceDiagBuilder::K_ImmediateWithCallStack;
}
