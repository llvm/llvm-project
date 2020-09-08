//===--- IgnoreExpr.cpp - Ignore intermediate Expressions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common functions to ignore intermediate expression nodes
//
//===----------------------------------------------------------------------===//

#include "clang/AST/IgnoreExpr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;

Expr *clang::IgnoreImplicitCastsSingleStep(Expr *E) {
  if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    return ICE->getSubExpr();

  if (auto *FE = dyn_cast<FullExpr>(E))
    return FE->getSubExpr();

  return E;
}

Expr *clang::IgnoreImplicitCastsExtraSingleStep(Expr *E) {
  // FIXME: Skip MaterializeTemporaryExpr and SubstNonTypeTemplateParmExpr in
  // addition to what IgnoreImpCasts() skips to account for the current
  // behaviour of IgnoreParenImpCasts().
  Expr *SubE = IgnoreImplicitCastsSingleStep(E);
  if (SubE != E)
    return SubE;

  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    return MTE->getSubExpr();

  if (auto *NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E))
    return NTTP->getReplacement();

  return E;
}

Expr *clang::IgnoreCastsSingleStep(Expr *E) {
  if (auto *CE = dyn_cast<CastExpr>(E))
    return CE->getSubExpr();

  if (auto *FE = dyn_cast<FullExpr>(E))
    return FE->getSubExpr();

  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    return MTE->getSubExpr();

  if (auto *NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E))
    return NTTP->getReplacement();

  return E;
}

Expr *clang::IgnoreLValueCastsSingleStep(Expr *E) {
  // Skip what IgnoreCastsSingleStep skips, except that only
  // lvalue-to-rvalue casts are skipped.
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() != CK_LValueToRValue)
      return E;

  return IgnoreCastsSingleStep(E);
}

Expr *clang::IgnoreBaseCastsSingleStep(Expr *E) {
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() == CK_DerivedToBase ||
        CE->getCastKind() == CK_UncheckedDerivedToBase ||
        CE->getCastKind() == CK_NoOp)
      return CE->getSubExpr();

  return E;
}

Expr *clang::IgnoreImplicitSingleStep(Expr *E) {
  Expr *SubE = IgnoreImplicitCastsSingleStep(E);
  if (SubE != E)
    return SubE;

  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    return MTE->getSubExpr();

  if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
    return BTE->getSubExpr();

  return E;
}

Expr *clang::IgnoreImplicitAsWrittenSingleStep(Expr *E) {
  if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    return ICE->getSubExprAsWritten();

  return IgnoreImplicitSingleStep(E);
}

Expr *clang::IgnoreParensOnlySingleStep(Expr *E) {
  if (auto *PE = dyn_cast<ParenExpr>(E))
    return PE->getSubExpr();
  return E;
}

Expr *clang::IgnoreParensSingleStep(Expr *E) {
  if (auto *PE = dyn_cast<ParenExpr>(E))
    return PE->getSubExpr();

  if (auto *UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_Extension)
      return UO->getSubExpr();
  }

  else if (auto *GSE = dyn_cast<GenericSelectionExpr>(E)) {
    if (!GSE->isResultDependent())
      return GSE->getResultExpr();
  }

  else if (auto *CE = dyn_cast<ChooseExpr>(E)) {
    if (!CE->isConditionDependent())
      return CE->getChosenSubExpr();
  }

  return E;
}
