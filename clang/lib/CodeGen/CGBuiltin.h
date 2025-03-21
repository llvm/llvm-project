//===------ CGBuiltin.h - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGBUILTIN_H
#define LLVM_CLANG_LIB_CODEGEN_CGBUILTIN_H

#include "CodeGenFunction.h"

using llvm::Value;
using llvm::Function;
using llvm::AtomicOrdering;
using clang::SmallVector;
using clang::CallExpr;
using clang::QualType;
using clang::CodeGen::Address;
using clang::CodeGen::CodeGenFunction;

// Many of MSVC builtins are on x64, ARM and AArch64; to avoid repeating code,
// we handle them here.
enum class clang::CodeGen::CodeGenFunction::MSVCIntrin {
  _BitScanForward,
  _BitScanReverse,
  _InterlockedAnd,
  _InterlockedCompareExchange,
  _InterlockedDecrement,
  _InterlockedExchange,
  _InterlockedExchangeAdd,
  _InterlockedExchangeSub,
  _InterlockedIncrement,
  _InterlockedOr,
  _InterlockedXor,
  _InterlockedExchangeAdd_acq,
  _InterlockedExchangeAdd_rel,
  _InterlockedExchangeAdd_nf,
  _InterlockedExchange_acq,
  _InterlockedExchange_rel,
  _InterlockedExchange_nf,
  _InterlockedCompareExchange_acq,
  _InterlockedCompareExchange_rel,
  _InterlockedCompareExchange_nf,
  _InterlockedCompareExchange128,
  _InterlockedCompareExchange128_acq,
  _InterlockedCompareExchange128_rel,
  _InterlockedCompareExchange128_nf,
  _InterlockedOr_acq,
  _InterlockedOr_rel,
  _InterlockedOr_nf,
  _InterlockedXor_acq,
  _InterlockedXor_rel,
  _InterlockedXor_nf,
  _InterlockedAnd_acq,
  _InterlockedAnd_rel,
  _InterlockedAnd_nf,
  _InterlockedIncrement_acq,
  _InterlockedIncrement_rel,
  _InterlockedIncrement_nf,
  _InterlockedDecrement_acq,
  _InterlockedDecrement_rel,
  _InterlockedDecrement_nf,
  __fastfail,
};

// Emit a simple intrinsic that has N scalar arguments and a return type
// matching the argument type. It is assumed that only the first argument is
// overloaded.
template <unsigned N>
Value *emitBuiltinWithOneOverloadedType(CodeGenFunction &CGF,
                                        const CallExpr *E,
                                        unsigned IntrinsicID,
                                        llvm::StringRef Name = "") {
  static_assert(N, "expect non-empty argument");
  SmallVector<Value *, N> Args;
  for (unsigned I = 0; I < N; ++I)
    Args.push_back(CGF.EmitScalarExpr(E->getArg(I)));
  Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Args[0]->getType());
  return CGF.Builder.CreateCall(F, Args, Name);
}

Value *emitUnaryMaybeConstrainedFPBuiltin(CodeGenFunction &CGF,
                                const CallExpr *E, unsigned IntrinsicID,
                                unsigned ConstrainedIntrinsicID);

Value *EmitToInt(CodeGenFunction &CGF, llvm::Value *V,
                 QualType T, llvm::IntegerType *IntType);

Value *EmitFromInt(CodeGenFunction &CGF, llvm::Value *V,
                   QualType T, llvm::Type *ResultType);

Address CheckAtomicAlignment(CodeGenFunction &CGF, const CallExpr *E);

Value *MakeBinaryAtomicValue(
    CodeGenFunction &CGF, llvm::AtomicRMWInst::BinOp Kind, const CallExpr *E,
    AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent);

Value *EmitOverflowIntrinsic(CodeGenFunction &CGF,
                             const llvm::Intrinsic::ID IntrinsicID,
                             Value *X, Value *Y, Value *&Carry);

Value *MakeAtomicCmpXchgValue(CodeGenFunction &CGF, const CallExpr *E,
                                     bool ReturnBool);

#endif
