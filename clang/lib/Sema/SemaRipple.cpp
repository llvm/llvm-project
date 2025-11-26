//===--- SemaRipple.cpp - Semantic Analysis for Ripple constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements semantic analysis for Ripple directives.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaRipple.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LLVM.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

using namespace clang;

#define DEBUG_TYPE "semaripple"

bool SemaRipple::CheckHasRippleBlockType(const Expr *E, unsigned BuiltinID) {
  auto *ENoCast = E->IgnoreParenImpCasts();
  bool HasValidRippleBlockShapeType = false;
  QualType PtrTy = ENoCast->getType().getDesugaredType(SemaRef.getASTContext());
  LLVM_DEBUG(
      llvm::dbgs() << "Type of BS\n\tExpr(";
      E->printPretty(llvm::dbgs(), nullptr, SemaRef.getPrintingPolicy());
      llvm::dbgs() << ")\n\tw/o parenthesis and impl casts\n\tExpr(";
      ENoCast->printPretty(llvm::dbgs(), nullptr, SemaRef.getPrintingPolicy());
      llvm::dbgs() << ")\n\tis\n\t" << PtrTy << "\n");
  if (PtrTy->isPointerType())
    if (const RecordType *RT = PtrTy->getPointeeType()->getAs<RecordType>())
      if (RT->getDecl()->getName() == "ripple_block_shape")
        HasValidRippleBlockShapeType = true;
  if (!HasValidRippleBlockShapeType) {
    StringRef OperationName = "ripple constructs";
    switch (BuiltinID) {
    case Builtin::BI__builtin_ripple_get_index:
      OperationName = "ripple_id";
      break;
    case Builtin::BI__builtin_ripple_get_size:
      OperationName = "ripple_get_block_size";
      break;
    case Builtin::BI__builtin_ripple_parallel_idx:
      OperationName = "ripple_parallel_idx";
      break;
    case Builtin::BI__builtin_ripple_broadcast_i8:
    case Builtin::BI__builtin_ripple_broadcast_u8:
    case Builtin::BI__builtin_ripple_broadcast_i16:
    case Builtin::BI__builtin_ripple_broadcast_u16:
    case Builtin::BI__builtin_ripple_broadcast_i32:
    case Builtin::BI__builtin_ripple_broadcast_u32:
    case Builtin::BI__builtin_ripple_broadcast_i64:
    case Builtin::BI__builtin_ripple_broadcast_u64:
    case Builtin::BI__builtin_ripple_broadcast_f16:
    case Builtin::BI__builtin_ripple_broadcast_bf16:
    case Builtin::BI__builtin_ripple_broadcast_f32:
    case Builtin::BI__builtin_ripple_broadcast_f64:
    case Builtin::BI__builtin_ripple_broadcast_p:
      OperationName = "ripple_broadcast";
      break;
    default:
      break;
    }
    SemaRef.Diag(E->getBeginLoc(), diag::err_ripple_block_shape_argument)
        << OperationName << PtrTy;
  }
  return !HasValidRippleBlockShapeType;
}

bool SemaRipple::CheckBuiltinFunctionCall(const FunctionDecl *FDecl,
                                          unsigned BuiltinID,
                                          const CallExpr *RippleBICall) {
  auto &ASTCtx = SemaRef.getASTContext();
  bool FoundErrors = false;
  switch (BuiltinID) {
  default:
    llvm_unreachable("Non-implemented check");
  case Builtin::BI__builtin_ripple_get_index:
  case Builtin::BI__builtin_ripple_get_size:
  case Builtin::BI__builtin_ripple_parallel_idx: {
    auto *BlockShapeArg = RippleBICall->getArg(0);
    if (CheckHasRippleBlockType(BlockShapeArg, BuiltinID))
      FoundErrors = true;

    int ArgNo = 2;
    // Arg
    for (auto *Arg : make_range(std::next(RippleBICall->arg_begin()),
                                RippleBICall->arg_end())) {
      Expr::EvalResult R;
      if (!Arg->getType()->isIntegralType(ASTCtx)) {
        SemaRef.Diag(RippleBICall->getBeginLoc(),
                     diag::err_builtin_invalid_arg_type)
            << ArgNo << /* scalar */ 1 << /* 'integer' ty */ 1 << /* no fp */ 0
            << Arg->getType() << Arg->getSourceRange();
        FoundErrors = true;
      } else {
        if (!Arg->isValueDependent() &&
            !Arg->EvaluateAsInt(R, SemaRef.getASTContext())) {
          SemaRef.Diag(RippleBICall->getBeginLoc(),
                       diag::err_constant_integer_arg_type)
              << FDecl->getName() << Arg->getSourceRange();
          FoundErrors = true;
        }
      }
      ArgNo++;
    }
  } break;
  }
  return FoundErrors;
}
