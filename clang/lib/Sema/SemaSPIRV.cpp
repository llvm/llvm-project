//===- SemaSPIRV.cpp - Semantic Analysis for SPIRV constructs--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SPIRV constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaSPIRV.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Sema/Sema.h"

namespace clang {

SemaSPIRV::SemaSPIRV(Sema &S) : SemaBase(S) {}

bool SemaSPIRV::CheckSPIRVBuiltinFunctionCall(unsigned BuiltinID,
                                              CallExpr *TheCall) {
  switch (BuiltinID) {
  case SPIRV::BI__builtin_spirv_distance: {
    return clang::Sema::CheckAllArgTypesAreCorrect(&SemaRef, TheCall, 2, 2);
  }
  case SPIRV::BI__builtin_spirv_length: {
    return clang::Sema::CheckAllArgTypesAreCorrect(&SemaRef, TheCall, 1, 2);
  }
  case SPIRV::BI__builtin_spirv_reflect: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    auto *VTyA = ArgTyA->getAs<VectorType>();
    if (VTyA == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyA
          << SemaRef.Context.getVectorType(ArgTyA, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    ExprResult B = TheCall->getArg(1);
    QualType ArgTyB = B.get()->getType();
    auto *VTyB = ArgTyB->getAs<VectorType>();
    if (VTyB == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyB
          << SemaRef.Context.getVectorType(ArgTyB, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  }
  return false;
}
} // namespace clang
