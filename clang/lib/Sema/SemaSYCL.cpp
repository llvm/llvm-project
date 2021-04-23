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
static QualType GetSYCLKernelObjectType(const FunctionDecl *KernelCaller) {
  assert(KernelCaller->getNumParams() > 0 && "Insufficient kernel parameters");
  QualType KernelParamTy = KernelCaller->getParamDecl(0)->getType();

  // SYCL 2020 kernels are passed by reference.
  if (KernelParamTy->isReferenceType())
    return KernelParamTy->getPointeeType();

  // SYCL 1.2.1
  return KernelParamTy;
}

void Sema::AddSYCLKernelLambda(const FunctionDecl *FD) {
  auto MangleCallback = [](ASTContext &Ctx,
                           const NamedDecl *ND) -> llvm::Optional<unsigned> {
    if (const auto *RD = dyn_cast<CXXRecordDecl>(ND))
      Ctx.AddSYCLKernelNamingDecl(RD);
    // We always want to go into the lambda mangling (skipping the unnamed
    // struct version), so make sure we return a value here.
    return 1;
  };

  QualType Ty = GetSYCLKernelObjectType(FD);
  std::unique_ptr<MangleContext> Ctx{ItaniumMangleContext::create(
      Context, Context.getDiagnostics(), MangleCallback)};
  llvm::raw_null_ostream Out;
  Ctx->mangleTypeName(Ty, Out);
}
