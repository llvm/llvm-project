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
    Sema *SemaPtr, CallExpr *TheCall,
    std::variant<QualType, std::nullopt_t> ExpectedType, CheckParam Check) {
  unsigned int NumElts;
  unsigned int expected;
  if (auto *n = std::get_if<PairParam>(&Check)) {
    if (SemaPtr->checkArgCount(TheCall, n->first)) {
      return true;
    }
    NumElts = n->first;
    expected = n->second;
  } else {
    NumElts = TheCall->getNumArgs();
  }

  for (unsigned i = 0; i < NumElts; i++) {
    Expr *localArg = TheCall->getArg(i);
    if (auto *val = std::get_if<QualType>(&ExpectedType)) {
      if (auto *fn = std::get_if<LLVMFnRef>(&Check)) {
        return CheckArgTypeIsCorrect(SemaPtr, localArg, *val, *fn);
      }
    }

    QualType PassedType = localArg->getType();
    if (PassedType->getAs<VectorType>() == nullptr) {
      SemaPtr->Diag(localArg->getBeginLoc(),
                    diag::err_typecheck_convert_incompatible)
          << PassedType
          << SemaPtr->Context.getVectorType(PassedType, expected,
                                            VectorKind::Generic)
          << 1 << 0 << 0;
      return true;
    }
  }

  if (std::get_if<PairParam>(&Check)) {
    if (auto *localArgVecTy =
            TheCall->getArg(0)->getType()->getAs<VectorType>()) {
      TheCall->setType(localArgVecTy->getElementType());
    }
  }

  return false;
}
