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

/// Checks if the first `NumArgsToCheck` arguments of a function call are of
/// vector type. If any of the arguments is not a vector type, it emits a
/// diagnostic error and returns `true`. Otherwise, it returns `false`.
///
/// \param TheCall The function call expression to check.
/// \param NumArgsToCheck The number of arguments to check for vector type.
/// \return `true` if any of the arguments is not a vector type, `false`
/// otherwise.

bool SemaSPIRV::CheckVectorArgs(CallExpr *TheCall, unsigned NumArgsToCheck) {
  for (unsigned i = 0; i < NumArgsToCheck; ++i) {
    ExprResult Arg = TheCall->getArg(i);
    QualType ArgTy = Arg.get()->getType();
    auto *VTy = ArgTy->getAs<VectorType>();
    if (VTy == nullptr) {
      SemaRef.Diag(Arg.get()->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTy
          << SemaRef.Context.getVectorType(ArgTy, 2, VectorKind::Generic) << 1
          << 0 << 0;
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

    // Use the helper function to check both arguments
    if (CheckVectorArgs(TheCall, 2))
      return true;

    QualType RetTy =
        TheCall->getArg(0)->getType()->getAs<VectorType>()->getElementType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_length: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    // Use the helper function to check the argument
    if (CheckVectorArgs(TheCall, 1))
      return true;

    QualType RetTy =
        TheCall->getArg(0)->getType()->getAs<VectorType>()->getElementType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_refract: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // Use the helper function to check the first two arguments
    if (CheckVectorArgs(TheCall, 2))
      return true;

    ExprResult C = TheCall->getArg(2);
    QualType ArgTyC = C.get()->getType();
    if (!ArgTyC->isFloatingType()) {
      SemaRef.Diag(C.get()->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << 3 << /* scalar*/ 5 << /* no int */ 0 << /* fp */ 1 << ArgTyC;
      return true;
    }

    QualType RetTy = TheCall->getArg(0)->getType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_reflect: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    // Use the helper function to check both arguments
    if (CheckVectorArgs(TheCall, 2))
      return true;

    QualType RetTy = TheCall->getArg(0)->getType();
    TheCall->setType(RetTy);
    break;
  }
  case SPIRV::BI__builtin_spirv_smoothstep: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    // check if the all arguments have floating representation
    for (unsigned i = 0; i < TheCall->getNumArgs(); ++i) {
      ExprResult Arg = TheCall->getArg(i);
      QualType ArgTy = Arg.get()->getType();
      if (!ArgTy->hasFloatingRepresentation()) {
        SemaRef.Diag(Arg.get()->getBeginLoc(),
                     diag::err_builtin_invalid_arg_type)
            << i + 1 << /* scalar or vector */ 5 << /* no int */ 0 << /* fp */ 1
            << ArgTy;
        return true;
      }
    }

    // check if all arguments are of the same type
    ExprResult A = TheCall->getArg(0);
    ExprResult B = TheCall->getArg(1);
    ExprResult C = TheCall->getArg(2);
    if (!(SemaRef.getASTContext().hasSameUnqualifiedType(A.get()->getType(),
                                                         B.get()->getType()) &&
          SemaRef.getASTContext().hasSameUnqualifiedType(A.get()->getType(),
                                                         C.get()->getType()))) {
      SemaRef.Diag(TheCall->getBeginLoc(),
                   diag::err_vec_builtin_incompatible_vector)
          << TheCall->getDirectCallee() << /*useAllTerminology*/ true
          << SourceRange(A.get()->getBeginLoc(), C.get()->getEndLoc());
      return true;
    }

    QualType RetTy = A.get()->getType();
    TheCall->setType(RetTy);
    break;
  }
  }
  return false;
}
} // namespace clang
