//===--- CGStmtRipple.cpp - Emit LLVM Code from Statements ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to emit Ripple intrinsics and statements as LLVM
// code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CGBuilder.h"
#include "CGValue.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CodeGenTypes.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRipple.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>
#include <string>

namespace llvm { class Type; }

using namespace clang;
using namespace CodeGen;
using namespace llvm;

namespace {

/**
 * Returns a size_t-typed llvm::Constant int from a clang AST expr. If `E`
 * cannot be mapped to a constant int, then a nullptr is returned.
 *
 */
Value *getConstSizetFromExpr(const Expr *E, CodeGenModule &CGM,
                             CGBuilderTy &Builder) {
  auto &Context = CGM.getContext();
  uint64_t sizeTNumBits = Context.getTypeSize(Context.getSizeType());

  Expr::EvalResult R;
  if (!E->EvaluateAsInt(R, Context))
    return nullptr;

  if (R.Val.getInt().isNegative()) {
    return nullptr;
  }

  return llvm::ConstantInt::get(
      Builder.getContext(),
      APInt(sizeTNumBits, R.Val.getInt().getExtValue(), false));
}

/**
 * Returns a uint64_t-typed llvm::Constant int from a clang AST expr. If `E`
 * cannot be mapped to a constant int, then a nullptr is returned.
 *
 */
Value *getConstU64FromExpr(const Expr *E, CodeGenModule &CGM,
                           CGBuilderTy &Builder) {

  Expr::EvalResult R;
  if (!E->EvaluateAsInt(R, CGM.getContext()))
    return nullptr;

  if (R.Val.getInt().isNegative()) {
    return nullptr;
  }

  return llvm::ConstantInt::get(Builder.getContext(),
                                APInt(64, R.Val.getInt().getExtValue(), false));
}

/**
 * Returns a int64_t-typed llvm::ConstantInt int from a clang AST expr. If `E`
 * cannot be mapped to a constant int, then a nullptr is returned.
 *
 */
Value *getConstI64FromExpr(const Expr *E, CodeGenModule &CGM,
                           CGBuilderTy &Builder) {

  Expr::EvalResult R;
  if (!E->EvaluateAsInt(R, CGM.getContext()))
    return nullptr;

  return llvm::ConstantInt::get(Builder.getContext(),
                                APInt(64, R.Val.getInt().getExtValue(), true));
}

/**
 * Returns the function pointer type corresponding to `size_t (*)(size_t)`.
 */
QualType getSizetToSizeTFuncPtr(ASTContext &Context) {
  QualType sizeTType = Context.getSizeType();
  FunctionProtoType::ExtProtoInfo EPI;
  EPI.ExtInfo = clang::FunctionType::ExtInfo();
  clang::QualType funcPtrType = Context.getPointerType(
      Context.getFunctionType(sizeTType, {sizeTType, sizeTType}, EPI));
  return funcPtrType;
}

#define CASE_RIPPLE_ALL_UNSIGNED_INT_BUILTIN(Name)                             \
  case Builtin::BI__builtin_ripple_##Name##_u8:                                \
  case Builtin::BI__builtin_ripple_##Name##_u16:                               \
  case Builtin::BI__builtin_ripple_##Name##_u32:                               \
  case Builtin::BI__builtin_ripple_##Name##_u64:

#define CASE_RIPPLE_ALL_SIGNED_INT_BUILTIN(Name)                               \
  case Builtin::BI__builtin_ripple_##Name##_i8:                                \
  case Builtin::BI__builtin_ripple_##Name##_i16:                               \
  case Builtin::BI__builtin_ripple_##Name##_i32:                               \
  case Builtin::BI__builtin_ripple_##Name##_i64:

#define CASE_RIPPLE_ALL_INT_BUILTIN(Name)                                      \
  CASE_RIPPLE_ALL_SIGNED_INT_BUILTIN(Name)                                     \
  CASE_RIPPLE_ALL_UNSIGNED_INT_BUILTIN(Name)

#define CASE_RIPPLE_ALL_FLOAT_BUILTIN(Name)                                    \
  case Builtin::BI__builtin_ripple_##Name##_f16:                               \
  case Builtin::BI__builtin_ripple_##Name##_bf16:                              \
  case Builtin::BI__builtin_ripple_##Name##_f32:                               \
  case Builtin::BI__builtin_ripple_##Name##_f64:

#define CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(Name)                                \
  CASE_RIPPLE_ALL_INT_BUILTIN(Name)                                            \
  CASE_RIPPLE_ALL_FLOAT_BUILTIN(Name)

/**
 * Returns the llvm ripple instrinsic ID for clang's ripple-specific BuiltinID
 * for reduce builtins.
 */
unsigned getReductionIntrinsicID(unsigned BuiltinID) {

  switch (BuiltinID) {

    CASE_RIPPLE_ALL_INT_BUILTIN(reduceadd)
    return llvm::Intrinsic::ripple_reduce_add;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reduceadd)
    return llvm::Intrinsic::ripple_reduce_fadd;

    CASE_RIPPLE_ALL_INT_BUILTIN(reducemul)
    return llvm::Intrinsic::ripple_reduce_mul;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reducemul)
    return llvm::Intrinsic::ripple_reduce_fmul;

    CASE_RIPPLE_ALL_SIGNED_INT_BUILTIN(reducemin)
    return llvm::Intrinsic::ripple_reduce_smin;

    CASE_RIPPLE_ALL_UNSIGNED_INT_BUILTIN(reducemin)
    return llvm::Intrinsic::ripple_reduce_umin;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reducemin)
    return llvm::Intrinsic::ripple_reduce_fmin;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reduceminimum)
    return llvm::Intrinsic::ripple_reduce_fminimum;

    CASE_RIPPLE_ALL_SIGNED_INT_BUILTIN(reducemax)
    return llvm::Intrinsic::ripple_reduce_smax;

    CASE_RIPPLE_ALL_UNSIGNED_INT_BUILTIN(reducemax)
    return llvm::Intrinsic::ripple_reduce_umax;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reducemax)
    return llvm::Intrinsic::ripple_reduce_fmax;

    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reducemaximum)
    return llvm::Intrinsic::ripple_reduce_fmaximum;

    CASE_RIPPLE_ALL_INT_BUILTIN(reduceand)
    return llvm::Intrinsic::ripple_reduce_and;

    CASE_RIPPLE_ALL_INT_BUILTIN(reduceor)
    return llvm::Intrinsic::ripple_reduce_or;

    CASE_RIPPLE_ALL_INT_BUILTIN(reducexor)
    return llvm::Intrinsic::ripple_reduce_xor;

  default:
    llvm_unreachable("Unexpected Builtin ID");
    return 0;
  }
}

/**
 * Returns the llvm ripple instrinsic ID for clang's ripple-specific BuiltinID
 * for saturation builtins.
 */
unsigned getSaturationIntrinsicID(unsigned BuiltinID) {

  switch (BuiltinID) {

  case Builtin::BI__builtin_ripple_add_sat_i8:
  case Builtin::BI__builtin_ripple_add_sat_i16:
  case Builtin::BI__builtin_ripple_add_sat_i32:
  case Builtin::BI__builtin_ripple_add_sat_i64:
    return llvm::Intrinsic::sadd_sat;
  case Builtin::BI__builtin_ripple_add_sat_u8:
  case Builtin::BI__builtin_ripple_add_sat_u16:
  case Builtin::BI__builtin_ripple_add_sat_u32:
  case Builtin::BI__builtin_ripple_add_sat_u64:
    return llvm::Intrinsic::uadd_sat;

  case Builtin::BI__builtin_ripple_sub_sat_i8:
  case Builtin::BI__builtin_ripple_sub_sat_i16:
  case Builtin::BI__builtin_ripple_sub_sat_i32:
  case Builtin::BI__builtin_ripple_sub_sat_i64:
    return llvm::Intrinsic::ssub_sat;
  case Builtin::BI__builtin_ripple_sub_sat_u8:
  case Builtin::BI__builtin_ripple_sub_sat_u16:
  case Builtin::BI__builtin_ripple_sub_sat_u32:
  case Builtin::BI__builtin_ripple_sub_sat_u64:
    return llvm::Intrinsic::usub_sat;

  case Builtin::BI__builtin_ripple_shl_sat_i8:
  case Builtin::BI__builtin_ripple_shl_sat_i16:
  case Builtin::BI__builtin_ripple_shl_sat_i32:
  case Builtin::BI__builtin_ripple_shl_sat_i64:
    return llvm::Intrinsic::sshl_sat;
  case Builtin::BI__builtin_ripple_shl_sat_u8:
  case Builtin::BI__builtin_ripple_shl_sat_u16:
  case Builtin::BI__builtin_ripple_shl_sat_u32:
  case Builtin::BI__builtin_ripple_shl_sat_u64:
    return llvm::Intrinsic::ushl_sat;

  default:
    llvm_unreachable("Unexpected Builtin ID");
    return 0;
  }
}

#define CASE_RIPPLE_MATH_BUILTIN(Name)                                         \
  case Builtin::BI__builtin_ripple_##Name##f16:                                \
  case Builtin::BI__builtin_ripple_##Name##f:                                  \
  case Builtin::BI__builtin_ripple_##Name:                                     \
  case Builtin::BI__builtin_ripple_##Name##l:

#define RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(Name)                         \
  CASE_RIPPLE_MATH_BUILTIN(Name)                                               \
  return llvm::Intrinsic::Name;

/**
 * Returns the llvm instrinsic ID for clang's ripple-specific BuiltinID
 * for math builtins.
 */
unsigned getMathIntrinsicId(unsigned BuiltinID) {

  switch (BuiltinID) {
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(sqrt)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(asin)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(acos)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(atan)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(atan2)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(sin)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(cos)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(tan)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(sinh)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(cosh)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(tanh)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(pow)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(log)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(log10)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(log2)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(exp)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(exp2)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(exp10)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(fabs)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(copysign)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(floor)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(ceil)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(trunc)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(rint)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(nearbyint)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(round)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(roundeven)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(sincos)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(modf)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(ldexp)
    RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC(frexp)

  default:
    llvm_unreachable("Unexpected Builtin ID");
    return 0;
  }
}

#undef RIPPLE_MATH_INTRINSICS_TO_LLVM_INTRINSIC

} // namespace

/**
 * Maps a call expression to one of ripple's builtins to llvm's ripple-specific
 * builtin.
 */
RValue CodeGenFunction::emitRippleBuiltin(const CallExpr *E,
                                          unsigned BuiltinID) {
  auto intrinsicsReturnType = [&]() -> llvm::Type * {
    return CGM.getTypes().ConvertType(E->getType());
  };
  SourceLocation Loc = E->getBeginLoc();
  StringRef CallName = E->getDirectCallee()->getIdentifier()->getName();

  switch (BuiltinID) {

  case Builtin::BI__builtin_ripple_parallel_idx:
    CGM.Error(Loc, "not within the scope of a ripple_parallel with matching "
                   "block shape (Block) and dimensions (Dims)");
    return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));

  /// --------------------------------------------------------------------------------------------
  /// PU-Specific Builtins Begin
  case Builtin::BI__builtin_ripple_get_index: {
    Value *BlockShape = EmitScalarExpr(E->getArg(0));
    Value *Dim = getConstSizetFromExpr(E->getArg(1), CGM, Builder);

    if (Dim == nullptr) {
      CGM.Error(
          Loc,
          "__builtin_ripple_get_index expects literal integer dimensions.");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }

    llvm::Function *F =
        CGM.getIntrinsic(llvm::Intrinsic::ripple_block_index, Dim->getType());
    llvm::Value *Result = Builder.CreateCall(F, {BlockShape, Dim});
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ripple_get_size: {
    Value *BlockShape = EmitScalarExpr(E->getArg(0));
    Value *Dim = getConstSizetFromExpr(E->getArg(1), CGM, Builder);

    if (Dim == nullptr) {
      CGM.Error(Loc,
                "__builtin_ripple_get_size expects literal integer arguments.");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }

    llvm::Function *F =
        CGM.getIntrinsic(llvm::Intrinsic::ripple_block_getsize, Dim->getType());
    llvm::Value *Result = Builder.CreateCall(F, {BlockShape, Dim});
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ripple_set_shape: {
    SmallVector<Value *, 11> IntrinsicArgs;
    assert(E->getNumArgs() == 11 &&
           "ripple set block shape takes 11 arguments");
    for (unsigned i = 0; i < E->getNumArgs(); ++i) {
      Value *Arg = getConstSizetFromExpr(E->getArg(i), CGM, Builder);
      if (Arg == nullptr) {
        CGM.Error(
            Loc,
            "__builtin_ripple_set_shape expects literal integer arguments.");
        return RValue::getIgnored();
      }
      IntrinsicArgs.push_back(Arg);
    }

    llvm::Function *F = CGM.getIntrinsic(llvm::Intrinsic::ripple_block_setshape,
                                         IntrinsicArgs[0]->getType());
    return RValue::get(Builder.CreateCall(F, IntrinsicArgs));
  }

    /// PU-Specific Builtins End
    /// --------------------------------------------------------------------------------------------
    /// Reduction-specific Builtings Begin
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(reduceadd)
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(reducemul)
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(reducemax)
    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reducemaximum)
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(reducemin)
    CASE_RIPPLE_ALL_FLOAT_BUILTIN(reduceminimum)
    CASE_RIPPLE_ALL_INT_BUILTIN(reduceand)
    CASE_RIPPLE_ALL_INT_BUILTIN(reduceor)
    CASE_RIPPLE_ALL_INT_BUILTIN(reducexor)

    {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);

      // Ripple semantics specifies the reassociativity of reductions
      IRBuilderBase::FastMathFlagGuard FMFG(Builder);
      Builder.getFastMathFlags().setAllowReassoc();

      Value *DimBitset = getConstU64FromExpr(E->getArg(0), CGM, Builder);
      Value *ReductionEl = EmitScalarExpr(E->getArg(1));

      if (DimBitset == nullptr) {
        CGM.Error(Loc,
                  CallName.str() +
                      " requires dimension literal bitmasks for dimensions.");
        return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
      }

      llvm::Function *F = CGM.getIntrinsic(getReductionIntrinsicID(BuiltinID),
                                           ReductionEl->getType());
      llvm::Value *Result = Builder.CreateCall(F, {DimBitset, ReductionEl});
      return RValue::get(Result);
    }
    /// Reduction-specific Builtins End
    /// --------------------------------------------------------------------------------------------
    /// Broadcast-specific Builtins Begin
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(broadcast)
  case Builtin::BI__builtin_ripple_broadcast_p: {
    Value *BlockShape = EmitScalarExpr(E->getArg(0));
    Value *Mask = getConstU64FromExpr(E->getArg(1), CGM, Builder);
    Value *ReductionEl = EmitScalarExpr(E->getArg(2));

    if (Mask == nullptr) {
      CGM.Error(Loc,
                CallName.str() +
                    " requires dimension literal bitmasks for dimensions.");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }

    llvm::Function *F = CGM.getIntrinsic(llvm::Intrinsic::ripple_broadcast,
                                         ReductionEl->getType());
    llvm::Value *Result =
        Builder.CreateCall(F, {BlockShape, Mask, ReductionEl});
    return RValue::get(Result);
  }
  /// Broadcast-specific Builtins End
  /// --------------------------------------------------------------------------------------------
  /// Slicing-specific Builtins Begin

#define RIPPLE_SLICE_MAX_ARGS 11
    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(slice)
  case Builtin::BI__builtin_ripple_slice_p: {
    SmallVector<Value *, RIPPLE_SLICE_MAX_ARGS> Args;
    Value *Slicee = EmitScalarExpr(E->getArg(0));
    Args.push_back(Slicee);
    for (int i = 1, n = E->getNumArgs(); i < n; ++i) {
      // TODO: Construct a ConstantExpr in Sema
      Value *constArg = getConstI64FromExpr(E->getArg(i), CGM, Builder);
      if (constArg == nullptr) {
        CGM.Error(Loc, "Non-constant slice argument to ripple_slice");
        return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
      }
      Args.push_back(constArg);
    }
    // LLVM intrinsics don't have variable number of args.
    // We limit the number of sliceable dimensions to RIPPLE_SLICE_MAX_ARGS -1.
    llvm::ConstantInt *MinusOne =
        llvm::ConstantInt::get(Builder.getContext(), APInt(64, -1, true));
    for (int i = E->getNumArgs(); i < RIPPLE_SLICE_MAX_ARGS; ++i) {
      Args.push_back(MinusOne);
    }

    llvm::Function *F =
        CGM.getIntrinsic(llvm::Intrinsic::ripple_slice, Slicee->getType());
    llvm::Value *Result = Builder.CreateCall(F, Args);
    return RValue::get(Result);
  }
#undef RIPPLE_SLICE_MAX_ARGS

    /// Broadcast-specific Builtins End
    /// --------------------------------------------------------------------------------------------
    /// Shuffle-specific Builtins Begin

    CASE_RIPPLE_ALL_INT_FLOAT_BUILTIN(shuffle)
  case Builtin::BI__builtin_ripple_shuffle_p: {
    // TODO: Migrate these checks to SemaRipple
    if (E->getNumArgs() != 4) {
      std::string ReturnT =
          E->getDirectCallee()->getFunctionType()->getReturnType().getAsString(
              CGM.getContext().getPrintingPolicy());
      CGM.Error(Loc, "Argument type mismatch for " + ReturnT + " " +
                         CallName.str() + "(" + ReturnT + ", " + ReturnT +
                         ", bool, " + " size_t (*)(size_t))." +
                         " Expected 4 arguments, got " +
                         std::to_string(E->getNumArgs()) + ".");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }

    QualType sizeTToSizeTFptr = getSizetToSizeTFuncPtr(CGM.getContext());
    QualType FnPtrArgT = E->getArg(3)->getType();

    if (!CGM.getContext().hasSameType(FnPtrArgT, sizeTToSizeTFptr)) {
      std::string ReturnT =
          E->getDirectCallee()->getFunctionType()->getReturnType().getAsString(
              CGM.getContext().getPrintingPolicy());
      CGM.Error(Loc, "Argument #4 type mismatch for " + ReturnT + " " +
                         CallName.str() + "(" + ReturnT + ", " + ReturnT +
                         ", bool, " + sizeTToSizeTFptr.getAsString() +
                         ". Expected size_t (*)(size_t, size_t), got " +
                         FnPtrArgT.getAsString() + ".");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }

    Value *ShuffleEl1 = EmitScalarExpr(E->getArg(0));
    Value *ShuffleEl2 = EmitScalarExpr(E->getArg(1));
    if (ShuffleEl1->getType() != ShuffleEl2->getType()) {
      CGM.Error(Loc, "Tensor operands of " + CallName.str() +
                         " must have the same type");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }
    bool DuoShuffle = false;
    if (!E->getArg(2)->EvaluateAsBooleanCondition(DuoShuffle,
                                                  CGM.getContext())) {
      std::string ReturnT =
          E->getDirectCallee()->getFunctionType()->getReturnType().getAsString(
              CGM.getContext().getPrintingPolicy());
      CGM.Error(Loc, "Argument #3 type mismatch for " + ReturnT + " " +
                         CallName.str() + "(" + ReturnT + ", " + ReturnT +
                         ", bool, " + sizeTToSizeTFptr.getAsString() +
                         ". Expected bool litteral, got " +
                         FnPtrArgT.getAsString() + ".");
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
    }
    Value *ShuffleFunc = EmitScalarExpr(E->getArg(3));

    llvm::Function *F =
        CGM.getIntrinsic(Intrinsic::ripple_shuffle, ShuffleEl1->getType());
    llvm::Value *Result = Builder.CreateCall(
        F, {ShuffleEl1, ShuffleEl2, Builder.getInt1(DuoShuffle), ShuffleFunc});
    return RValue::get(Result);
  }
    /// Shuffle-specific Builtins End
    /// --------------------------------------------------------------------------------------------
    /// Saturation-specific Builtins Start
    CASE_RIPPLE_ALL_INT_BUILTIN(add_sat)
    CASE_RIPPLE_ALL_INT_BUILTIN(sub_sat)
    CASE_RIPPLE_ALL_INT_BUILTIN(shl_sat) {
      Value *X = EmitScalarExpr(E->getArg(0));
      Value *Y = EmitScalarExpr(E->getArg(1));

      llvm::Function *F =
          CGM.getIntrinsic(getSaturationIntrinsicID(BuiltinID), {X->getType()});
      llvm::Value *Result = Builder.CreateCall(F, {X, Y});
      return RValue::get(Result);
    }

    /// Saturation-specific Builtins End

    /// Math-specific Builtins Begin
    /// Unary builtins
    CASE_RIPPLE_MATH_BUILTIN(sqrt)
    CASE_RIPPLE_MATH_BUILTIN(asin)
    CASE_RIPPLE_MATH_BUILTIN(acos)
    CASE_RIPPLE_MATH_BUILTIN(atan)
    CASE_RIPPLE_MATH_BUILTIN(atan2)
    CASE_RIPPLE_MATH_BUILTIN(sin)
    CASE_RIPPLE_MATH_BUILTIN(cos)
    CASE_RIPPLE_MATH_BUILTIN(tan)
    CASE_RIPPLE_MATH_BUILTIN(sinh)
    CASE_RIPPLE_MATH_BUILTIN(cosh)
    CASE_RIPPLE_MATH_BUILTIN(tanh)
    CASE_RIPPLE_MATH_BUILTIN(pow)
    CASE_RIPPLE_MATH_BUILTIN(log)
    CASE_RIPPLE_MATH_BUILTIN(log10)
    CASE_RIPPLE_MATH_BUILTIN(log2)
    CASE_RIPPLE_MATH_BUILTIN(exp)
    CASE_RIPPLE_MATH_BUILTIN(exp2)
    CASE_RIPPLE_MATH_BUILTIN(exp10)
    CASE_RIPPLE_MATH_BUILTIN(fabs)
    CASE_RIPPLE_MATH_BUILTIN(copysign)
    CASE_RIPPLE_MATH_BUILTIN(floor)
    CASE_RIPPLE_MATH_BUILTIN(ceil)
    CASE_RIPPLE_MATH_BUILTIN(trunc)
    CASE_RIPPLE_MATH_BUILTIN(rint)
    CASE_RIPPLE_MATH_BUILTIN(nearbyint)
    CASE_RIPPLE_MATH_BUILTIN(round)
    CASE_RIPPLE_MATH_BUILTIN(roundeven) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      SmallVector<Value *, 2> Args;
      for (auto *Arg : E->arguments())
        Args.push_back(EmitScalarExpr(Arg));

      // Most llvm math intrinsics (even binary) only mangle with one float type
      llvm::Function *F =
          CGM.getIntrinsic(getMathIntrinsicId(BuiltinID), {Args[0]->getType()});
      llvm::Value *Result = Builder.CreateCall(F, Args);
      return RValue::get(Result);
    }
    CASE_RIPPLE_MATH_BUILTIN(ldexp) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      SmallVector<Value *, 2> Args;
      for (auto *Arg : E->arguments())
        Args.push_back(EmitScalarExpr(Arg));
      llvm::Function *F =
          CGM.getIntrinsic(getMathIntrinsicId(BuiltinID),
                           {Args[0]->getType(), Args[1]->getType()});
      llvm::Value *Result = Builder.CreateCall(F, Args);
      return RValue::get(Result);
    }
    CASE_RIPPLE_MATH_BUILTIN(sincos) {
      // The llvm intinsic has one argument and two return values, i.e.,
      // [sin, cos] = llvm.sincos.fX(val)
      // Our intinsic has one input floating point value, a pointer to a
      // floating point used to store the sin, a pointer to a floating point
      // used to store the cos, and returns void.
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Value *Arg = EmitScalarExpr(E->getArg(0));
      llvm::Function *F =
          CGM.getIntrinsic(getMathIntrinsicId(BuiltinID), {Arg->getType()});
      llvm::Value *SinCosVal = Builder.CreateCall(F, {Arg});

      Address SinAddr = EmitPointerWithAlignment(E->getArg(1));
      llvm::Value *SinValue = Builder.CreateExtractValue(SinCosVal, 0, "sin");
      Builder.CreateStore(SinValue, SinAddr);

      Address CosAddr = EmitPointerWithAlignment(E->getArg(2));
      llvm::Value *CosValue = Builder.CreateExtractValue(SinCosVal, 1, "cos");
      Builder.CreateStore(CosValue, CosAddr);

      return RValue::getIgnored();
    }
    CASE_RIPPLE_MATH_BUILTIN(modf) {
      // The llvm intinsic has one argument and two return values
      // Our intinsic has one input floating point value, one pointer to a
      // floating point value storing the integral part, and returns a floating
      // point value representing the fractional part
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Value *Arg = EmitScalarExpr(E->getArg(0));
      llvm::Function *F =
          CGM.getIntrinsic(getMathIntrinsicId(BuiltinID), {Arg->getType()});
      llvm::Value *ModfVal = Builder.CreateCall(F, {Arg});

      Address IntegralPartAddr = EmitPointerWithAlignment(E->getArg(1));
      llvm::Value *IntegralPartVal =
          Builder.CreateExtractValue(ModfVal, 0, "integral");
      Builder.CreateStore(IntegralPartVal, IntegralPartAddr);

      llvm::Value *FractionalPartVal =
          Builder.CreateExtractValue(ModfVal, 1, "fractional");
      return RValue::get(FractionalPartVal);
    }
    CASE_RIPPLE_MATH_BUILTIN(frexp) {
      // The llvm intinsic has one argument and two return values
      // Our intinsic has one input floating point value, one pointer to an
      // integer representing the exponent, and returns a floating point value
      // representing the fractional part
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Address IntegralPartAddr = EmitPointerWithAlignment(E->getArg(1));

      Value *Arg = EmitScalarExpr(E->getArg(0));
      llvm::Function *F =
          CGM.getIntrinsic(getMathIntrinsicId(BuiltinID),
                           {Arg->getType(), IntegralPartAddr.getElementType()});
      llvm::Value *ModfVal = Builder.CreateCall(F, {Arg});

      llvm::Value *IntegralPartVal =
          Builder.CreateExtractValue(ModfVal, 1, "integral");
      Builder.CreateStore(IntegralPartVal, IntegralPartAddr);

      llvm::Value *FractionalPartVal =
          Builder.CreateExtractValue(ModfVal, 0, "fractional");
      return RValue::get(FractionalPartVal);
    }
  default:
    CGM.Error(Loc, "Unexpected Ripple builtin: " + CallName.str());
    if (E->getType()->isVoidType())
      return RValue::getIgnored();
    else
      return RValue::get(llvm::PoisonValue::get(intrinsicsReturnType()));
  }
}

#undef CASE_RIPPLE_MATH_BUILTIN
#undef CASE_RIPPLE_ALLINT_ALLFLOAT_BUILTIN
#undef CASE_RIPPLE_ALLFLOAT_BUILTIN
#undef CASE_RIPPLE_ALLINT_BUILTIN

void CodeGenFunction::EmitRippleComputeConstruct(
    const RippleComputeConstruct &S, ArrayRef<const Attr *> Attrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("ripple.par.for.end");

  LexicalScope RippleParallelForScope(
      *this, S.getAssociatedForStmt()->getSourceRange());

  // Make it clear that we are entering a ripple parallel loop
  llvm::BasicBlock *ParallelLoopEntry =
      createBasicBlock("ripple.par.for.begin");
  EmitBranch(ParallelLoopEntry);
  EmitBlock(ParallelLoopEntry);

  // All local variable used by ripple, and the original induction variable
  // if a DeclStmt was present in the associated loop's init
  for (const auto *VD : S.getRippleVarDecls())
    EmitVarDecl(*VD);

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();

  // Emit the ripple parallel loop of full blocks
  if (!S.getRippleLoopStmt()) {
    CGM.Error(S.getBeginLoc(),
              "ripple_parallel construct has no loop statement");
    return;
  }
  EmitForStmt(*S.getRippleLoopStmt(), Attrs);

  // Update the loop IV for the last time
  // When the remainder is empty, the induction variable has for value the upper
  // bound (we added the step NumIteration times), otherwise we re-compute the
  // upper bound at the end of the remainder body.
  EmitStmt(S.getLoopIVUpdate());

  if (S.generateRemainder()) {
    llvm::BasicBlock *ForBodyRippleCond =
        createBasicBlock("ripple.par.for.remainder.cond");

    Stmt::Likelihood LH = Stmt::LH_None;
    uint64_t RemainderCount = getProfileCount(S.getRemainderBody());
    if (!RemainderCount && !getCurrentProfileCount() &&
        CGM.getCodeGenOpts().OptimizationLevel)
      LH = Stmt::getLikelihood(S.getRemainderBody(), nullptr);
    if (!CGM.getCodeGenOpts().MCDCCoverage) {
      EmitBranchOnBoolExpr(S.getRemainderRuntimeCond(), ForBodyRippleCond,
                           ExitBlock, RemainderCount, LH,
                           /*ConditionalOp=*/nullptr);
    } else {
      llvm::Value *BoolCondVal =
          EvaluateExprAsBool(S.getRemainderRuntimeCond());
      Builder.CreateCondBr(BoolCondVal, ForBodyRippleCond, ExitBlock);
    }

    // Remainder tensor condition
    EmitBlock(ForBodyRippleCond);

    llvm::BasicBlock *ForBodyRemainder =
        createBasicBlock("ripple.par.for.remainder.body");

    if (!CGM.getCodeGenOpts().MCDCCoverage) {
      EmitBranchOnBoolExpr(S.getAssociatedForStmt()->getCond(),
                           ForBodyRemainder, ExitBlock, RemainderCount, LH,
                           /*ConditionalOp=*/nullptr);
    } else {
      llvm::Value *BoolCondVal =
          EvaluateExprAsBool(S.getAssociatedForStmt()->getCond());
      Builder.CreateCondBr(BoolCondVal, ForBodyRemainder, ExitBlock);
    }

    // It's the last iteration, continue goes to the exit!
    BreakContinueStack.push_back(
        BreakContinue(*S.getAssociatedForStmt(), LoopExit, LoopExit));

    EmitBlock(ForBodyRemainder);
    {
      RunCleanupsScope BodyScope(*this);
      if (!S.getRemainderBody()) {
        CGM.Error(S.getBeginLoc(),
                  "ripple_parallel construct has no remainder body statement");
        return;
      }
      EmitStmt(S.getRemainderBody());
    }
    EmitStmt(S.getEndLoopIVUpdate());

    BreakContinueStack.pop_back();
  }

  EmitBlock(ExitBlock,
            /*IsFinished*/ !RippleParallelForScope.requiresCleanups());
}
