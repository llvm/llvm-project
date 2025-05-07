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

static bool CheckAllArgsHaveSameType(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() > 1);
  QualType ArgTy0 = TheCall->getArg(0)->getType();

  for (unsigned I = 1, N = TheCall->getNumArgs(); I < N; ++I) {
    if (!S->getASTContext().hasSameUnqualifiedType(
            ArgTy0, TheCall->getArg(I)->getType())) {
      S->Diag(TheCall->getBeginLoc(), diag::err_vec_builtin_incompatible_vector)
          << TheCall->getDirectCallee() << /*useAllTerminology*/ true
          << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                         TheCall->getArg(N - 1)->getEndLoc());
      return true;
    }
  }
  return false;
}

bool SemaSPIRV::CheckSPIRVBuiltinFunctionCall(unsigned BuiltinID,
                                              CallExpr *TheCall) {
  switch (BuiltinID) {
  case SPIRV::BI__builtin_spirv_distance: {
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

    QualType RetTy = VTyA->getElementType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_length: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    auto *VTy = ArgTyA->getAs<VectorType>();
    if (VTy == nullptr) {
      SemaRef.Diag(A.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyA
          << SemaRef.Context.getVectorType(ArgTyA, 2, VectorKind::Generic) << 1
          << 0 << 0;
      return true;
    }
    QualType RetTy = VTy->getElementType();
    TheCall->setType(RetTy);
    break;
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
  case SPIRV::BI__builtin_spirv_smoothstep: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // Check if first argument has floating representation
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    if (!ArgTyA->hasFloatingRepresentation()) {
      SemaRef.Diag(A.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
          << /* fp */ 1 << ArgTyA;
      return true;
    }

    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_faceforward: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // Check if first argument has floating representation
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    if (!ArgTyA->hasFloatingRepresentation()) {
      SemaRef.Diag(A.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
          << /* fp */ 1 << ArgTyA;
      return true;
    }

    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    QualType RetTy = ArgTyA;
    TheCall->setType(RetTy);
    break;
  }
  }
  return false;
}
} // namespace clang
