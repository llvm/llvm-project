//===--------- DirectX.cpp - Emit LLVM Code for builtins ------------------===//
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

#include "CGHLSLRuntime.h"
#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

Value *CodeGenFunction::EmitDirectXBuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {
  switch (BuiltinID) {
  case DirectX::BI__builtin_dx_dot2add: {
    Value *A = EmitScalarExpr(E->getArg(0));
    Value *B = EmitScalarExpr(E->getArg(1));
    Value *Acc = EmitScalarExpr(E->getArg(2));

    Value *AX = Builder.CreateExtractElement(A, Builder.getSize(0));
    Value *AY = Builder.CreateExtractElement(A, Builder.getSize(1));
    Value *BX = Builder.CreateExtractElement(B, Builder.getSize(0));
    Value *BY = Builder.CreateExtractElement(B, Builder.getSize(1));

    Intrinsic::ID ID = llvm ::Intrinsic::dx_dot2add;
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Acc->getType(), ID,
        ArrayRef<Value *>{Acc, AX, AY, BX, BY}, nullptr, "dx.dot2add");
  }
  }
  return nullptr;
}
