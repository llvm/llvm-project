//===-- WebAssembly.cpp - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGBuiltin.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

Value *CodeGenFunction::EmitWebAssemblyBuiltinExpr(unsigned BuiltinID,
                                                   const CallExpr *E) {
  switch (BuiltinID) {
  case WebAssembly::BI__builtin_wasm_memory_size: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *I = EmitScalarExpr(E->getArg(0));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_memory_size, ResultType);
    return Builder.CreateCall(Callee, I);
  }
  case WebAssembly::BI__builtin_wasm_memory_grow: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Args[] = {EmitScalarExpr(E->getArg(0)),
                     EmitScalarExpr(E->getArg(1))};
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_memory_grow, ResultType);
    return Builder.CreateCall(Callee, Args);
  }
  case WebAssembly::BI__builtin_wasm_tls_size: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_tls_size, ResultType);
    return Builder.CreateCall(Callee);
  }
  case WebAssembly::BI__builtin_wasm_tls_align: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_tls_align, ResultType);
    return Builder.CreateCall(Callee);
  }
  case WebAssembly::BI__builtin_wasm_tls_base: {
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_tls_base);
    return Builder.CreateCall(Callee);
  }
  case WebAssembly::BI__builtin_wasm_throw: {
    Value *Tag = EmitScalarExpr(E->getArg(0));
    Value *Obj = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_throw);
    return EmitRuntimeCallOrInvoke(Callee, {Tag, Obj});
  }
  case WebAssembly::BI__builtin_wasm_rethrow: {
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_rethrow);
    return EmitRuntimeCallOrInvoke(Callee);
  }
  case WebAssembly::BI__builtin_wasm_memory_atomic_wait32: {
    Value *Addr = EmitScalarExpr(E->getArg(0));
    Value *Expected = EmitScalarExpr(E->getArg(1));
    Value *Timeout = EmitScalarExpr(E->getArg(2));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_memory_atomic_wait32);
    return Builder.CreateCall(Callee, {Addr, Expected, Timeout});
  }
  case WebAssembly::BI__builtin_wasm_memory_atomic_wait64: {
    Value *Addr = EmitScalarExpr(E->getArg(0));
    Value *Expected = EmitScalarExpr(E->getArg(1));
    Value *Timeout = EmitScalarExpr(E->getArg(2));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_memory_atomic_wait64);
    return Builder.CreateCall(Callee, {Addr, Expected, Timeout});
  }
  case WebAssembly::BI__builtin_wasm_memory_atomic_notify: {
    Value *Addr = EmitScalarExpr(E->getArg(0));
    Value *Count = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_memory_atomic_notify);
    return Builder.CreateCall(Callee, {Addr, Count});
  }
  case WebAssembly::BI__builtin_wasm_trunc_s_i32_f32:
  case WebAssembly::BI__builtin_wasm_trunc_s_i32_f64:
  case WebAssembly::BI__builtin_wasm_trunc_s_i64_f32:
  case WebAssembly::BI__builtin_wasm_trunc_s_i64_f64: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Type *ResT = ConvertType(E->getType());
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_trunc_signed, {ResT, Src->getType()});
    return Builder.CreateCall(Callee, {Src});
  }
  case WebAssembly::BI__builtin_wasm_trunc_u_i32_f32:
  case WebAssembly::BI__builtin_wasm_trunc_u_i32_f64:
  case WebAssembly::BI__builtin_wasm_trunc_u_i64_f32:
  case WebAssembly::BI__builtin_wasm_trunc_u_i64_f64: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Type *ResT = ConvertType(E->getType());
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_trunc_unsigned,
                                        {ResT, Src->getType()});
    return Builder.CreateCall(Callee, {Src});
  }
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i32_f32:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i32_f64:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i64_f32:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i64_f64:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i16x8_f16x8:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_s_i32x4_f32x4: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Type *ResT = ConvertType(E->getType());
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::fptosi_sat, {ResT, Src->getType()});
    return Builder.CreateCall(Callee, {Src});
  }
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i32_f32:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i32_f64:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i64_f32:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i64_f64:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i16x8_f16x8:
  case WebAssembly::BI__builtin_wasm_trunc_saturate_u_i32x4_f32x4: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Type *ResT = ConvertType(E->getType());
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::fptoui_sat, {ResT, Src->getType()});
    return Builder.CreateCall(Callee, {Src});
  }
  case WebAssembly::BI__builtin_wasm_min_f32:
  case WebAssembly::BI__builtin_wasm_min_f64:
  case WebAssembly::BI__builtin_wasm_min_f16x8:
  case WebAssembly::BI__builtin_wasm_min_f32x4:
  case WebAssembly::BI__builtin_wasm_min_f64x2: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::minimum, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_max_f32:
  case WebAssembly::BI__builtin_wasm_max_f64:
  case WebAssembly::BI__builtin_wasm_max_f16x8:
  case WebAssembly::BI__builtin_wasm_max_f32x4:
  case WebAssembly::BI__builtin_wasm_max_f64x2: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::maximum, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_pmin_f16x8:
  case WebAssembly::BI__builtin_wasm_pmin_f32x4:
  case WebAssembly::BI__builtin_wasm_pmin_f64x2: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_pmin, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_pmax_f16x8:
  case WebAssembly::BI__builtin_wasm_pmax_f32x4:
  case WebAssembly::BI__builtin_wasm_pmax_f64x2: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_pmax, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_ceil_f16x8:
  case WebAssembly::BI__builtin_wasm_floor_f16x8:
  case WebAssembly::BI__builtin_wasm_trunc_f16x8:
  case WebAssembly::BI__builtin_wasm_nearest_f16x8:
  case WebAssembly::BI__builtin_wasm_ceil_f32x4:
  case WebAssembly::BI__builtin_wasm_floor_f32x4:
  case WebAssembly::BI__builtin_wasm_trunc_f32x4:
  case WebAssembly::BI__builtin_wasm_nearest_f32x4:
  case WebAssembly::BI__builtin_wasm_ceil_f64x2:
  case WebAssembly::BI__builtin_wasm_floor_f64x2:
  case WebAssembly::BI__builtin_wasm_trunc_f64x2:
  case WebAssembly::BI__builtin_wasm_nearest_f64x2: {
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_ceil_f16x8:
    case WebAssembly::BI__builtin_wasm_ceil_f32x4:
    case WebAssembly::BI__builtin_wasm_ceil_f64x2:
      IntNo = Intrinsic::ceil;
      break;
    case WebAssembly::BI__builtin_wasm_floor_f16x8:
    case WebAssembly::BI__builtin_wasm_floor_f32x4:
    case WebAssembly::BI__builtin_wasm_floor_f64x2:
      IntNo = Intrinsic::floor;
      break;
    case WebAssembly::BI__builtin_wasm_trunc_f16x8:
    case WebAssembly::BI__builtin_wasm_trunc_f32x4:
    case WebAssembly::BI__builtin_wasm_trunc_f64x2:
      IntNo = Intrinsic::trunc;
      break;
    case WebAssembly::BI__builtin_wasm_nearest_f16x8:
    case WebAssembly::BI__builtin_wasm_nearest_f32x4:
    case WebAssembly::BI__builtin_wasm_nearest_f64x2:
      IntNo = Intrinsic::nearbyint;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Value *Value = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(IntNo, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, Value);
  }
  case WebAssembly::BI__builtin_wasm_ref_null_extern: {
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_ref_null_extern);
    return Builder.CreateCall(Callee);
  }
  case WebAssembly::BI__builtin_wasm_ref_is_null_extern: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_ref_is_null_extern);
    return Builder.CreateCall(Callee, {Src});
  }
  case WebAssembly::BI__builtin_wasm_ref_null_func: {
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_ref_null_func);
    return Builder.CreateCall(Callee);
  }
  case WebAssembly::BI__builtin_wasm_test_function_pointer_signature: {
    Value *FuncRef = EmitScalarExpr(E->getArg(0));

    // Get the function type from the argument's static type
    QualType ArgType = E->getArg(0)->getType();
    const PointerType *PtrTy = ArgType->getAs<PointerType>();
    assert(PtrTy && "Sema should have ensured this is a function pointer");

    const FunctionType *FuncTy = PtrTy->getPointeeType()->getAs<FunctionType>();
    assert(FuncTy && "Sema should have ensured this is a function pointer");

    // In the llvm IR, we won't have access any more to the type of the function
    // pointer so we need to insert this type information somehow. The
    // @llvm.wasm.ref.test.func takes varargs arguments whose values are unused
    // to indicate the type of the function to test for. See the test here:
    // llvm/test/CodeGen/WebAssembly/ref-test-func.ll
    //
    // The format is: first we include the return types (since this is a C
    // function pointer, there will be 0 or one of these) then a token type to
    // indicate the boundary between return types and param types, then the
    // param types.

    llvm::FunctionType *LLVMFuncTy =
        cast<llvm::FunctionType>(ConvertType(QualType(FuncTy, 0)));

    bool VarArg = LLVMFuncTy->isVarArg();
    unsigned NParams = LLVMFuncTy->getNumParams();
    std::vector<Value *> Args;
    Args.reserve(NParams + 3 + VarArg);
    // The only real argument is the FuncRef
    Args.push_back(FuncRef);

    // Add the type information
    llvm::Type *RetType = LLVMFuncTy->getReturnType();
    if (!RetType->isVoidTy()) {
      Args.push_back(PoisonValue::get(RetType));
    }
    // The token type indicates the boundary between return types and param
    // types.
    Args.push_back(PoisonValue::get(llvm::Type::getTokenTy(getLLVMContext())));
    for (unsigned i = 0; i < NParams; i++) {
      Args.push_back(PoisonValue::get(LLVMFuncTy->getParamType(i)));
    }
    if (VarArg) {
      Args.push_back(PoisonValue::get(Builder.getPtrTy()));
    }
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_ref_test_func);
    return Builder.CreateCall(Callee, Args);
  }
  case WebAssembly::BI__builtin_wasm_swizzle_i8x16: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    Value *Indices = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_swizzle);
    return Builder.CreateCall(Callee, {Src, Indices});
  }
  case WebAssembly::BI__builtin_wasm_abs_i8x16:
  case WebAssembly::BI__builtin_wasm_abs_i16x8:
  case WebAssembly::BI__builtin_wasm_abs_i32x4:
  case WebAssembly::BI__builtin_wasm_abs_i64x2: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    Value *Neg = Builder.CreateNeg(Vec, "neg");
    Constant *Zero = llvm::Constant::getNullValue(Vec->getType());
    Value *ICmp = Builder.CreateICmpSLT(Vec, Zero, "abscond");
    return Builder.CreateSelect(ICmp, Neg, Vec, "abs");
  }
  case WebAssembly::BI__builtin_wasm_avgr_u_i8x16:
  case WebAssembly::BI__builtin_wasm_avgr_u_i16x8: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_avgr_unsigned,
                                        ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_q15mulr_sat_s_i16x8: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_q15mulr_sat_signed);
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_extadd_pairwise_i8x16_s_i16x8:
  case WebAssembly::BI__builtin_wasm_extadd_pairwise_i8x16_u_i16x8:
  case WebAssembly::BI__builtin_wasm_extadd_pairwise_i16x8_s_i32x4:
  case WebAssembly::BI__builtin_wasm_extadd_pairwise_i16x8_u_i32x4: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_extadd_pairwise_i8x16_s_i16x8:
    case WebAssembly::BI__builtin_wasm_extadd_pairwise_i16x8_s_i32x4:
      IntNo = Intrinsic::wasm_extadd_pairwise_signed;
      break;
    case WebAssembly::BI__builtin_wasm_extadd_pairwise_i8x16_u_i16x8:
    case WebAssembly::BI__builtin_wasm_extadd_pairwise_i16x8_u_i32x4:
      IntNo = Intrinsic::wasm_extadd_pairwise_unsigned;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }

    Function *Callee = CGM.getIntrinsic(IntNo, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, Vec);
  }
  case WebAssembly::BI__builtin_wasm_bitselect: {
    Value *V1 = EmitScalarExpr(E->getArg(0));
    Value *V2 = EmitScalarExpr(E->getArg(1));
    Value *C = EmitScalarExpr(E->getArg(2));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_bitselect, ConvertType(E->getType()));
    return Builder.CreateCall(Callee, {V1, V2, C});
  }
  case WebAssembly::BI__builtin_wasm_dot_s_i32x4_i16x8: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_dot);
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_any_true_v128:
  case WebAssembly::BI__builtin_wasm_all_true_i8x16:
  case WebAssembly::BI__builtin_wasm_all_true_i16x8:
  case WebAssembly::BI__builtin_wasm_all_true_i32x4:
  case WebAssembly::BI__builtin_wasm_all_true_i64x2: {
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_any_true_v128:
      IntNo = Intrinsic::wasm_anytrue;
      break;
    case WebAssembly::BI__builtin_wasm_all_true_i8x16:
    case WebAssembly::BI__builtin_wasm_all_true_i16x8:
    case WebAssembly::BI__builtin_wasm_all_true_i32x4:
    case WebAssembly::BI__builtin_wasm_all_true_i64x2:
      IntNo = Intrinsic::wasm_alltrue;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Value *Vec = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(IntNo, Vec->getType());
    return Builder.CreateCall(Callee, {Vec});
  }
  case WebAssembly::BI__builtin_wasm_bitmask_i8x16:
  case WebAssembly::BI__builtin_wasm_bitmask_i16x8:
  case WebAssembly::BI__builtin_wasm_bitmask_i32x4:
  case WebAssembly::BI__builtin_wasm_bitmask_i64x2: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_bitmask, Vec->getType());
    return Builder.CreateCall(Callee, {Vec});
  }
  case WebAssembly::BI__builtin_wasm_abs_f16x8:
  case WebAssembly::BI__builtin_wasm_abs_f32x4:
  case WebAssembly::BI__builtin_wasm_abs_f64x2: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(Intrinsic::fabs, Vec->getType());
    return Builder.CreateCall(Callee, {Vec});
  }
  case WebAssembly::BI__builtin_wasm_sqrt_f16x8:
  case WebAssembly::BI__builtin_wasm_sqrt_f32x4:
  case WebAssembly::BI__builtin_wasm_sqrt_f64x2: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(Intrinsic::sqrt, Vec->getType());
    return Builder.CreateCall(Callee, {Vec});
  }
  case WebAssembly::BI__builtin_wasm_narrow_s_i8x16_i16x8:
  case WebAssembly::BI__builtin_wasm_narrow_u_i8x16_i16x8:
  case WebAssembly::BI__builtin_wasm_narrow_s_i16x8_i32x4:
  case WebAssembly::BI__builtin_wasm_narrow_u_i16x8_i32x4: {
    Value *Low = EmitScalarExpr(E->getArg(0));
    Value *High = EmitScalarExpr(E->getArg(1));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_narrow_s_i8x16_i16x8:
    case WebAssembly::BI__builtin_wasm_narrow_s_i16x8_i32x4:
      IntNo = Intrinsic::wasm_narrow_signed;
      break;
    case WebAssembly::BI__builtin_wasm_narrow_u_i8x16_i16x8:
    case WebAssembly::BI__builtin_wasm_narrow_u_i16x8_i32x4:
      IntNo = Intrinsic::wasm_narrow_unsigned;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Function *Callee =
        CGM.getIntrinsic(IntNo, {ConvertType(E->getType()), Low->getType()});
    return Builder.CreateCall(Callee, {Low, High});
  }
  case WebAssembly::BI__builtin_wasm_trunc_sat_s_zero_f64x2_i32x4:
  case WebAssembly::BI__builtin_wasm_trunc_sat_u_zero_f64x2_i32x4: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_trunc_sat_s_zero_f64x2_i32x4:
      IntNo = Intrinsic::fptosi_sat;
      break;
    case WebAssembly::BI__builtin_wasm_trunc_sat_u_zero_f64x2_i32x4:
      IntNo = Intrinsic::fptoui_sat;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    llvm::Type *SrcT = Vec->getType();
    llvm::Type *TruncT = SrcT->getWithNewType(Builder.getInt32Ty());
    Function *Callee = CGM.getIntrinsic(IntNo, {TruncT, SrcT});
    Value *Trunc = Builder.CreateCall(Callee, Vec);
    Value *Splat = Constant::getNullValue(TruncT);
    return Builder.CreateShuffleVector(Trunc, Splat, {0, 1, 2, 3});
  }
  case WebAssembly::BI__builtin_wasm_shuffle_i8x16: {
    Value *Ops[18];
    size_t OpIdx = 0;
    Ops[OpIdx++] = EmitScalarExpr(E->getArg(0));
    Ops[OpIdx++] = EmitScalarExpr(E->getArg(1));
    while (OpIdx < 18) {
      std::optional<llvm::APSInt> LaneConst =
          E->getArg(OpIdx)->getIntegerConstantExpr(getContext());
      assert(LaneConst && "Constant arg isn't actually constant?");
      Ops[OpIdx++] = llvm::ConstantInt::get(getLLVMContext(), *LaneConst);
    }
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_shuffle);
    return Builder.CreateCall(Callee, Ops);
  }
  case WebAssembly::BI__builtin_wasm_relaxed_madd_f16x8:
  case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f16x8:
  case WebAssembly::BI__builtin_wasm_relaxed_madd_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_madd_f64x2:
  case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f64x2: {
    Value *A = EmitScalarExpr(E->getArg(0));
    Value *B = EmitScalarExpr(E->getArg(1));
    Value *C = EmitScalarExpr(E->getArg(2));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_relaxed_madd_f16x8:
    case WebAssembly::BI__builtin_wasm_relaxed_madd_f32x4:
    case WebAssembly::BI__builtin_wasm_relaxed_madd_f64x2:
      IntNo = Intrinsic::wasm_relaxed_madd;
      break;
    case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f16x8:
    case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f32x4:
    case WebAssembly::BI__builtin_wasm_relaxed_nmadd_f64x2:
      IntNo = Intrinsic::wasm_relaxed_nmadd;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Function *Callee = CGM.getIntrinsic(IntNo, A->getType());
    return Builder.CreateCall(Callee, {A, B, C});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_laneselect_i8x16:
  case WebAssembly::BI__builtin_wasm_relaxed_laneselect_i16x8:
  case WebAssembly::BI__builtin_wasm_relaxed_laneselect_i32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_laneselect_i64x2: {
    Value *A = EmitScalarExpr(E->getArg(0));
    Value *B = EmitScalarExpr(E->getArg(1));
    Value *C = EmitScalarExpr(E->getArg(2));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_relaxed_laneselect, A->getType());
    return Builder.CreateCall(Callee, {A, B, C});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_swizzle_i8x16: {
    Value *Src = EmitScalarExpr(E->getArg(0));
    Value *Indices = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_relaxed_swizzle);
    return Builder.CreateCall(Callee, {Src, Indices});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_min_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_max_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_min_f64x2:
  case WebAssembly::BI__builtin_wasm_relaxed_max_f64x2: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_relaxed_min_f32x4:
    case WebAssembly::BI__builtin_wasm_relaxed_min_f64x2:
      IntNo = Intrinsic::wasm_relaxed_min;
      break;
    case WebAssembly::BI__builtin_wasm_relaxed_max_f32x4:
    case WebAssembly::BI__builtin_wasm_relaxed_max_f64x2:
      IntNo = Intrinsic::wasm_relaxed_max;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Function *Callee = CGM.getIntrinsic(IntNo, LHS->getType());
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_trunc_s_i32x4_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_trunc_u_i32x4_f32x4:
  case WebAssembly::BI__builtin_wasm_relaxed_trunc_s_zero_i32x4_f64x2:
  case WebAssembly::BI__builtin_wasm_relaxed_trunc_u_zero_i32x4_f64x2: {
    Value *Vec = EmitScalarExpr(E->getArg(0));
    unsigned IntNo;
    switch (BuiltinID) {
    case WebAssembly::BI__builtin_wasm_relaxed_trunc_s_i32x4_f32x4:
      IntNo = Intrinsic::wasm_relaxed_trunc_signed;
      break;
    case WebAssembly::BI__builtin_wasm_relaxed_trunc_u_i32x4_f32x4:
      IntNo = Intrinsic::wasm_relaxed_trunc_unsigned;
      break;
    case WebAssembly::BI__builtin_wasm_relaxed_trunc_s_zero_i32x4_f64x2:
      IntNo = Intrinsic::wasm_relaxed_trunc_signed_zero;
      break;
    case WebAssembly::BI__builtin_wasm_relaxed_trunc_u_zero_i32x4_f64x2:
      IntNo = Intrinsic::wasm_relaxed_trunc_unsigned_zero;
      break;
    default:
      llvm_unreachable("unexpected builtin ID");
    }
    Function *Callee = CGM.getIntrinsic(IntNo);
    return Builder.CreateCall(Callee, {Vec});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_q15mulr_s_i16x8: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_relaxed_q15mulr_signed);
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_dot_i8x16_i7x16_s_i16x8: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_relaxed_dot_i8x16_i7x16_signed);
    return Builder.CreateCall(Callee, {LHS, RHS});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_dot_i8x16_i7x16_add_s_i32x4: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Value *Acc = EmitScalarExpr(E->getArg(2));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_relaxed_dot_i8x16_i7x16_add_signed);
    return Builder.CreateCall(Callee, {LHS, RHS, Acc});
  }
  case WebAssembly::BI__builtin_wasm_relaxed_dot_bf16x8_add_f32_f32x4: {
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));
    Value *Acc = EmitScalarExpr(E->getArg(2));
    Function *Callee =
        CGM.getIntrinsic(Intrinsic::wasm_relaxed_dot_bf16x8_add_f32);
    return Builder.CreateCall(Callee, {LHS, RHS, Acc});
  }
  case WebAssembly::BI__builtin_wasm_loadf16_f32: {
    Value *Addr = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_loadf16_f32);
    return Builder.CreateCall(Callee, {Addr});
  }
  case WebAssembly::BI__builtin_wasm_storef16_f32: {
    Value *Val = EmitScalarExpr(E->getArg(0));
    Value *Addr = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_storef16_f32);
    return Builder.CreateCall(Callee, {Val, Addr});
  }
  case WebAssembly::BI__builtin_wasm_splat_f16x8: {
    Value *Val = EmitScalarExpr(E->getArg(0));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_splat_f16x8);
    return Builder.CreateCall(Callee, {Val});
  }
  case WebAssembly::BI__builtin_wasm_extract_lane_f16x8: {
    Value *Vector = EmitScalarExpr(E->getArg(0));
    Value *Index = EmitScalarExpr(E->getArg(1));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_extract_lane_f16x8);
    return Builder.CreateCall(Callee, {Vector, Index});
  }
  case WebAssembly::BI__builtin_wasm_replace_lane_f16x8: {
    Value *Vector = EmitScalarExpr(E->getArg(0));
    Value *Index = EmitScalarExpr(E->getArg(1));
    Value *Val = EmitScalarExpr(E->getArg(2));
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_replace_lane_f16x8);
    return Builder.CreateCall(Callee, {Vector, Index, Val});
  }
  case WebAssembly::BI__builtin_wasm_table_get: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *Table = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Value *Index = EmitScalarExpr(E->getArg(1));
    Function *Callee;
    if (E->getType().isWebAssemblyExternrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_get_externref);
    else if (E->getType().isWebAssemblyFuncrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_get_funcref);
    else
      llvm_unreachable(
          "Unexpected reference type for __builtin_wasm_table_get");
    return Builder.CreateCall(Callee, {Table, Index});
  }
  case WebAssembly::BI__builtin_wasm_table_set: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *Table = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Value *Index = EmitScalarExpr(E->getArg(1));
    Value *Val = EmitScalarExpr(E->getArg(2));
    Function *Callee;
    if (E->getArg(2)->getType().isWebAssemblyExternrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_set_externref);
    else if (E->getArg(2)->getType().isWebAssemblyFuncrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_set_funcref);
    else
      llvm_unreachable(
          "Unexpected reference type for __builtin_wasm_table_set");
    return Builder.CreateCall(Callee, {Table, Index, Val});
  }
  case WebAssembly::BI__builtin_wasm_table_size: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *Value = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_table_size);
    return Builder.CreateCall(Callee, Value);
  }
  case WebAssembly::BI__builtin_wasm_table_grow: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *Table = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Value *Val = EmitScalarExpr(E->getArg(1));
    Value *NElems = EmitScalarExpr(E->getArg(2));

    Function *Callee;
    if (E->getArg(1)->getType().isWebAssemblyExternrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_grow_externref);
    else if (E->getArg(2)->getType().isWebAssemblyFuncrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_fill_funcref);
    else
      llvm_unreachable(
          "Unexpected reference type for __builtin_wasm_table_grow");

    return Builder.CreateCall(Callee, {Table, Val, NElems});
  }
  case WebAssembly::BI__builtin_wasm_table_fill: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *Table = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Value *Index = EmitScalarExpr(E->getArg(1));
    Value *Val = EmitScalarExpr(E->getArg(2));
    Value *NElems = EmitScalarExpr(E->getArg(3));

    Function *Callee;
    if (E->getArg(2)->getType().isWebAssemblyExternrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_fill_externref);
    else if (E->getArg(2)->getType().isWebAssemblyFuncrefType())
      Callee = CGM.getIntrinsic(Intrinsic::wasm_table_fill_funcref);
    else
      llvm_unreachable(
          "Unexpected reference type for __builtin_wasm_table_fill");

    return Builder.CreateCall(Callee, {Table, Index, Val, NElems});
  }
  case WebAssembly::BI__builtin_wasm_table_copy: {
    assert(E->getArg(0)->getType()->isArrayType());
    Value *TableX = EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(*this);
    Value *TableY = EmitArrayToPointerDecay(E->getArg(1)).emitRawPointer(*this);
    Value *DstIdx = EmitScalarExpr(E->getArg(2));
    Value *SrcIdx = EmitScalarExpr(E->getArg(3));
    Value *NElems = EmitScalarExpr(E->getArg(4));

    Function *Callee = CGM.getIntrinsic(Intrinsic::wasm_table_copy);

    return Builder.CreateCall(Callee, {TableX, TableY, SrcIdx, DstIdx, NElems});
  }
  default:
    return nullptr;
  }
}
