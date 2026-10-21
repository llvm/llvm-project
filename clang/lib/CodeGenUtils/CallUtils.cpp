//===--- CallUtils.cpp - Shared call emission queries ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/CallUtils.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/TargetCXXABI.h"

namespace clang::CodeGenUtils {

CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *MD) {
  return MD->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

llvm::FPClassTest getNoFPClassTestMask(const LangOptions &LangOpts) {
  llvm::FPClassTest Mask = llvm::fcNone;
  if (LangOpts.NoHonorInfs)
    Mask |= llvm::fcInf;
  if (LangOpts.NoHonorNaNs)
    Mask |= llvm::fcNan;
  return Mask;
}

EvaluationOrder getCallArgEvaluationOrder(const CallExpr *E) {
  // Note that we allow this to override the order dictated by the calling
  // convention on the MS ABI, which means that parameter destruction order is
  // not necessarily reverse construction order.
  // FIXME: Revisit this based on C++ committee response to unimplementability.
  const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E);
  if (!OCE)
    return EvaluationOrder::Default;

  if (OCE->isAssignmentOp())
    return EvaluationOrder::ForceRightToLeft;

  switch (OCE->getOperator()) {
  case OO_LessLess:
  case OO_GreaterGreater:
  case OO_AmpAmp:
  case OO_PipePipe:
  case OO_Comma:
  case OO_ArrowStar:
    return EvaluationOrder::ForceLeftToRight;
  default:
    return EvaluationOrder::Default;
  }
}

bool shouldEvaluateArgsLeftToRight(EvaluationOrder Order,
                                   const TargetCXXABI &ABI) {
  return ABI.areArgsDestroyedLeftToRightInCallee()
             ? Order == EvaluationOrder::ForceLeftToRight
             : Order != EvaluationOrder::ForceRightToLeft;
}

} // namespace clang::CodeGenUtils
