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

#include "CGBuiltin.h"
#include "CGHLSLRuntime.h"
#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

// Has second type mangled argument.
static Value *
emitBinaryExpMaybeConstrainedFPBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                                       Intrinsic::ID IntrinsicID,
                                       Intrinsic::ID ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID,
                                       {Src0->getType(), Src1->getType()});
    return CGF.Builder.CreateConstrainedFPCall(F, {Src0, Src1});
  }

  Function *F =
      CGF.CGM.getIntrinsic(IntrinsicID, {Src0->getType(), Src1->getType()});
  return CGF.Builder.CreateCall(F, {Src0, Src1});
}

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
  case SPIRV::BI__builtin_spirv_generic_cast_to_ptr_explicit: {
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasPointerRepresentation() &&
           E->getArg(1)->getType()->hasIntegerRepresentation() &&
           "GenericCastToPtrExplicit takes a pointer and an int");
    llvm::Type *Res = getTypes().ConvertType(E->getType());
    assert(Res->isPointerTy() &&
           "GenericCastToPtrExplicit doesn't return a pointer");
    llvm::CallInst *Call = Builder.CreateIntrinsic(
        /*ReturnType=*/Res, Intrinsic::spv_generic_cast_to_ptr_explicit,
        ArrayRef<Value *>{Ptr}, nullptr, "spv.generic_cast");
    Call->addRetAttr(llvm::Attribute::AttrKind::NoUndef);
    return Call;
  }
  case Builtin::BIlogbf:
  case Builtin::BI__builtin_logbf: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *FrExpFunc = CGM.getIntrinsic(
        Intrinsic::frexp, {Src0->getType(), Builder.getInt32Ty()});
    CallInst *FrExp = Builder.CreateCall(FrExpFunc, Src0);
    Value *Exp = Builder.CreateExtractValue(FrExp, 1);
    Value *Add = Builder.CreateAdd(
        Exp, ConstantInt::getSigned(Exp->getType(), -1), "", false, true);
    Value *SIToFP = Builder.CreateSIToFP(Add, Builder.getFloatTy());
    Value *Fabs =
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs);
    Value *FCmpONE = Builder.CreateFCmpONE(
        Fabs, ConstantFP::getInfinity(Builder.getFloatTy()));
    Value *Sel1 = Builder.CreateSelect(FCmpONE, SIToFP, Fabs);
    Value *FCmpOEQ =
        Builder.CreateFCmpOEQ(Src0, ConstantFP::getZero(Builder.getFloatTy()));
    Value *Sel2 = Builder.CreateSelect(
        FCmpOEQ,
        ConstantFP::getInfinity(Builder.getFloatTy(), /*Negative=*/true), Sel1);
    return Sel2;
  }
  case Builtin::BIlogb:
  case Builtin::BI__builtin_logb: {
    Value *Src0 = EmitScalarExpr(E->getArg(0));
    Function *FrExpFunc = CGM.getIntrinsic(
        Intrinsic::frexp, {Src0->getType(), Builder.getInt32Ty()});
    CallInst *FrExp = Builder.CreateCall(FrExpFunc, Src0);
    Value *Exp = Builder.CreateExtractValue(FrExp, 1);
    Value *Add = Builder.CreateAdd(
        Exp, ConstantInt::getSigned(Exp->getType(), -1), "", false, true);
    Value *SIToFP = Builder.CreateSIToFP(Add, Builder.getDoubleTy());
    Value *Fabs =
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs);
    Value *FCmpONE = Builder.CreateFCmpONE(
        Fabs, ConstantFP::getInfinity(Builder.getDoubleTy()));
    Value *Sel1 = Builder.CreateSelect(FCmpONE, SIToFP, Fabs);
    Value *FCmpOEQ =
        Builder.CreateFCmpOEQ(Src0, ConstantFP::getZero(Builder.getDoubleTy()));
    Value *Sel2 = Builder.CreateSelect(
        FCmpOEQ,
        ConstantFP::getInfinity(Builder.getDoubleTy(), /*Negative=*/true),
        Sel1);
    return Sel2;
  }
  case Builtin::BIscalbnf:
  case Builtin::BI__builtin_scalbnf:
  case Builtin::BIscalbn:
  case Builtin::BI__builtin_scalbn:
    return emitBinaryExpMaybeConstrainedFPBuiltin(
        *this, E, Intrinsic::ldexp, Intrinsic::experimental_constrained_ldexp);
  }
  return nullptr;
}
