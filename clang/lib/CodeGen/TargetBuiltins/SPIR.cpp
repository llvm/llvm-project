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

#include "CGHLSLRuntime.h"
#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
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
  case SPIRV::BI__builtin_spirv_smoothstep: {
    Value *Min = EmitScalarExpr(E->getArg(0));
    Value *Max = EmitScalarExpr(E->getArg(1));
    Value *X = EmitScalarExpr(E->getArg(2));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           E->getArg(2)->getType()->hasFloatingRepresentation() &&
           "SmoothStep operands must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Min->getType(), Intrinsic::spv_smoothstep,
        ArrayRef<Value *>{Min, Max, X}, /*FMFSource=*/nullptr,
        "spv.smoothstep");
  }
  case SPIRV::BI__builtin_spirv_faceforward: {
    Value *N = EmitScalarExpr(E->getArg(0));
    Value *I = EmitScalarExpr(E->getArg(1));
    Value *Ng = EmitScalarExpr(E->getArg(2));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           E->getArg(2)->getType()->hasFloatingRepresentation() &&
           "FaceForward operands must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/N->getType(), Intrinsic::spv_faceforward,
        ArrayRef<Value *>{N, I, Ng}, /*FMFSource=*/nullptr, "spv.faceforward");
  }
  }
  return nullptr;
}
