//===--- Common.cpp --- Semantic Analysis common implementation file ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements common functions used in SPIRV and HLSL semantic
// analysis constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/Common.h"

bool clang::CheckArgTypeIsCorrect(
    Sema *S, Expr *Arg, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check) {
  QualType PassedType = Arg->getType();
  if (Check(PassedType)) {
    if (auto *VecTyA = PassedType->getAs<VectorType>())
      ExpectedType = S->Context.getVectorType(
          ExpectedType, VecTyA->getNumElements(), VecTyA->getVectorKind());
    S->Diag(Arg->getBeginLoc(), diag::err_typecheck_convert_incompatible)
        << PassedType << ExpectedType << 1 << 0 << 0;
    return true;
  }
  return false;
}

bool clang::CheckAllArgTypesAreCorrect(
    Sema *S, CallExpr *TheCall, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check) {
  for (unsigned i = 0; i < TheCall->getNumArgs(); ++i) {
    Expr *Arg = TheCall->getArg(i);
    if (CheckArgTypeIsCorrect(S, Arg, ExpectedType, Check)) {
      return true;
    }
  }
  return false;
}

bool clang::CheckAllArgTypesAreCorrect(Sema *SemaPtr, CallExpr *TheCall,
                                       unsigned int NumOfElts,
                                       unsigned int expectedNumOfElts) {
  if (SemaPtr->checkArgCount(TheCall, NumOfElts)) {
    return true;
  }

  for (unsigned i = 0; i < NumOfElts; i++) {
    Expr *localArg = TheCall->getArg(i);
    QualType PassedType = localArg->getType();
    QualType ExpectedType = SemaPtr->Context.getVectorType(
        PassedType, expectedNumOfElts, VectorKind::Generic);
    auto Check = [](QualType PassedType) {
      return PassedType->getAs<VectorType>() == nullptr;
    };

    if (CheckArgTypeIsCorrect(SemaPtr, localArg, ExpectedType, Check)) {
      return true;
    }
  }

  if (auto *localArgVecTy =
          TheCall->getArg(0)->getType()->getAs<VectorType>()) {
    TheCall->setType(localArgVecTy->getElementType());
  }

  return false;
}
