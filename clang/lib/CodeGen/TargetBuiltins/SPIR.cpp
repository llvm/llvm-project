//===--------- SPIR.cpp - Emit LLVM Code for builtins ---------------------===//
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

#include "ABIInfo.h"
#include "CGHLSLRuntime.h"
#include "CodeGenFunction.h"
#include "TargetInfo.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

Value *CodeGenFunction::EmitSPIRVBuiltinExpr(unsigned BuiltinID,
                                             const CallExpr *E) {
  switch (BuiltinID) {
  case SPIRV::BI__builtin_spirv_distance: {
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "Distance operands must have a float representation");
    assert(E->getArg(0)->getType()->isVectorType() &&
           E->getArg(1)->getType()->isVectorType() &&
           "Distance operands must be a vector");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType()->getScalarType(), Intrinsic::spv_distance,
        ArrayRef<Value *>{X, Y}, nullptr, "spv.distance");
  }
  case SPIRV::BI__builtin_spirv_length: {
    Value *X = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "length operand must have a float representation");
    assert(E->getArg(0)->getType()->isVectorType() &&
           "length operand must be a vector");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType()->getScalarType(), Intrinsic::spv_length,
        ArrayRef<Value *>{X}, nullptr, "spv.length");
  }
  case SPIRV::BI__builtin_spirv_reflect: {
    Value *I = EmitScalarExpr(E->getArg(0));
    Value *N = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "Reflect operands must have a float representation");
    assert(E->getArg(0)->getType()->isVectorType() &&
           E->getArg(1)->getType()->isVectorType() &&
           "Reflect operands must be a vector");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/I->getType(), Intrinsic::spv_reflect,
        ArrayRef<Value *>{I, N}, nullptr, "spv.reflect");
  }
  }
  return nullptr;
}
