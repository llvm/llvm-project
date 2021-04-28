//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/AST/Mangle.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

Sema::SemaDiagnosticBuilder Sema::SYCLDiagIfDeviceCode(SourceLocation Loc,
                                                       unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(getCurLexicalContext());
  SemaDiagnosticBuilder::Kind DiagKind = [this, FD] {
    if (!FD)
      return SemaDiagnosticBuilder::K_Nop;
    if (getEmissionStatus(FD) == Sema::FunctionEmissionStatus::Emitted)
      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    return SemaDiagnosticBuilder::K_Deferred;
  }();
  return SemaDiagnosticBuilder(DiagKind, Loc, DiagID, FD, *this);
}

bool Sema::checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  assert(Callee && "Callee may not be null.");

  // Errors in unevaluated context don't need to be generated,
  // so we can safely skip them.
  if (isUnevaluatedContext() || isConstantEvaluated())
    return true;

  SemaDiagnosticBuilder::Kind DiagKind = SemaDiagnosticBuilder::K_Nop;

  return DiagKind != SemaDiagnosticBuilder::K_Immediate &&
         DiagKind != SemaDiagnosticBuilder::K_ImmediateWithCallStack;
}

// The SYCL kernel's 'object type' used for diagnostics and naming/mangling is
// the first parameter to a sycl_kernel labeled function template. In SYCL1.2.1,
// this was passed by value, and in SYCL2020, it is passed by reference.
static const CXXRecordDecl *
GetSYCLKernelObjectDecl(const FunctionDecl *KernelCaller) {
  assert(KernelCaller->getNumParams() > 0 && "Insufficient kernel parameters");
  QualType KernelParamTy = KernelCaller->getParamDecl(0)->getType();

  // SYCL 2020 kernels are passed by reference.
  if (KernelParamTy->isReferenceType())
    return KernelParamTy->getPointeeCXXRecordDecl();

  // SYCL 1.2.1
  return KernelParamTy->getAsCXXRecordDecl();
}

void Sema::AddSYCLKernelLambda(const FunctionDecl *FD) {
  auto ShouldMangleCallback = [](ASTContext &Ctx, const TagDecl *TD) {
    // We ALWAYS want to descend into the lambda mangling for these.
    return true;
  };
  auto MangleCallback = [](ASTContext &Ctx, const TagDecl *TD, raw_ostream &) {
    Ctx.AddSYCLKernelNamingDecl(TD);
  };

  const CXXRecordDecl *KernelObjectDecl = GetSYCLKernelObjectDecl(FD);
  QualType Ty{KernelObjectDecl->getTypeForDecl(), 0};
  std::unique_ptr<MangleContext> Ctx{ItaniumMangleContext::create(
      Context, Context.getDiagnostics(), ShouldMangleCallback, MangleCallback)};
  llvm::raw_null_ostream Out;
  Ctx->mangleTypeName(Ty, Out);
  (void)FD;
}
