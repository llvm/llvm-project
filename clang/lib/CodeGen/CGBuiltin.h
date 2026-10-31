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

// Fills in the trailing parameters of an intrinsic that the builtin does not
// expose, using the values declared via ImmArg<..., DefaultValue<...>>.
void appendDefaultIntrinsicArgs(llvm::SmallVectorImpl<llvm::Value *> &Args,
                                llvm::Function *F);

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
                                    const clang::CallExpr *E, bool ReturnBool,
                                    llvm::AtomicOrdering SuccessOrdering,
                                    llvm::AtomicOrdering FailureOrdering);

#endif
