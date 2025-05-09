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
llvm::Value *emitBuiltinWithOneOverloadedType(clang::CodeGen::CodeGenFunction &CGF,
                                              const clang::CallExpr *E,
                                              unsigned IntrinsicID,
                                              llvm::StringRef Name = "") {
  static_assert(N, "expect non-empty argument");
  clang::SmallVector<llvm::Value *, N> Args;
  for (unsigned I = 0; I < N; ++I)
    Args.push_back(CGF.EmitScalarExpr(E->getArg(I)));
  llvm::Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Args[0]->getType());
  return CGF.Builder.CreateCall(F, Args, Name);
}

llvm::Value *emitUnaryMaybeConstrainedFPBuiltin(clang::CodeGen::CodeGenFunction &CGF,
                                                const clang::CallExpr *E,
                                                unsigned IntrinsicID,
                                                unsigned ConstrainedIntrinsicID);

llvm::Value *EmitToInt(clang::CodeGen::CodeGenFunction &CGF, llvm::Value *V,
                       clang::QualType T, llvm::IntegerType *IntType);

llvm::Value *EmitFromInt(clang::CodeGen::CodeGenFunction &CGF, llvm::Value *V,
                         clang::QualType T, llvm::Type *ResultType);

clang::CodeGen::Address CheckAtomicAlignment(clang::CodeGen::CodeGenFunction &CGF,
                                             const clang::CallExpr *E);

llvm::Value *MakeBinaryAtomicValue(clang::CodeGen::CodeGenFunction &CGF,
                                   llvm::AtomicRMWInst::BinOp Kind,
                                   const clang::CallExpr *E,
                                   llvm::AtomicOrdering Ordering =
                                      llvm::AtomicOrdering::SequentiallyConsistent);

llvm::Value *EmitOverflowIntrinsic(clang::CodeGen::CodeGenFunction &CGF,
                                   const llvm::Intrinsic::ID IntrinsicID,
                                   llvm::Value *X,
                                   llvm::Value *Y,
                                   llvm::Value *&Carry);

llvm::Value *MakeAtomicCmpXchgValue(clang::CodeGen::CodeGenFunction &CGF,
                                    const clang::CallExpr *E,
                                    bool ReturnBool);

#endif
