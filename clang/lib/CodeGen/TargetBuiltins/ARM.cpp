//===---------- ARM.cpp - Emit LLVM Code for builtins ---------------------===//
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
#include "CGBuiltin.h"
#include "CGDebugInfo.h"
#include "TargetInfo.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/TargetParser/AArch64TargetParser.h"

#include <numeric>

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static std::optional<CodeGenFunction::MSVCIntrin>
translateAarch64ToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CodeGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::AArch64::BI_BitScanForward:
  case clang::AArch64::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::AArch64::BI_BitScanReverse:
  case clang::AArch64::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::AArch64::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::AArch64::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::AArch64::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::AArch64::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::AArch64::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::AArch64::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::AArch64::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::AArch64::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  case clang::AArch64::BI_InterlockedExchangeAdd8_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd16_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd64_acq:
    return MSVCIntrin::_InterlockedExchangeAdd_acq;
  case clang::AArch64::BI_InterlockedExchangeAdd8_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd16_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd64_rel:
    return MSVCIntrin::_InterlockedExchangeAdd_rel;
  case clang::AArch64::BI_InterlockedExchangeAdd8_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd16_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd64_nf:
    return MSVCIntrin::_InterlockedExchangeAdd_nf;
  case clang::AArch64::BI_InterlockedExchange8_acq:
  case clang::AArch64::BI_InterlockedExchange16_acq:
  case clang::AArch64::BI_InterlockedExchange_acq:
  case clang::AArch64::BI_InterlockedExchange64_acq:
  case clang::AArch64::BI_InterlockedExchangePointer_acq:
    return MSVCIntrin::_InterlockedExchange_acq;
  case clang::AArch64::BI_InterlockedExchange8_rel:
  case clang::AArch64::BI_InterlockedExchange16_rel:
  case clang::AArch64::BI_InterlockedExchange_rel:
  case clang::AArch64::BI_InterlockedExchange64_rel:
  case clang::AArch64::BI_InterlockedExchangePointer_rel:
    return MSVCIntrin::_InterlockedExchange_rel;
  case clang::AArch64::BI_InterlockedExchange8_nf:
  case clang::AArch64::BI_InterlockedExchange16_nf:
  case clang::AArch64::BI_InterlockedExchange_nf:
  case clang::AArch64::BI_InterlockedExchange64_nf:
  case clang::AArch64::BI_InterlockedExchangePointer_nf:
    return MSVCIntrin::_InterlockedExchange_nf;
  case clang::AArch64::BI_InterlockedCompareExchange8_acq:
  case clang::AArch64::BI_InterlockedCompareExchange16_acq:
  case clang::AArch64::BI_InterlockedCompareExchange_acq:
  case clang::AArch64::BI_InterlockedCompareExchange64_acq:
  case clang::AArch64::BI_InterlockedCompareExchangePointer_acq:
    return MSVCIntrin::_InterlockedCompareExchange_acq;
  case clang::AArch64::BI_InterlockedCompareExchange8_rel:
  case clang::AArch64::BI_InterlockedCompareExchange16_rel:
  case clang::AArch64::BI_InterlockedCompareExchange_rel:
  case clang::AArch64::BI_InterlockedCompareExchange64_rel:
  case clang::AArch64::BI_InterlockedCompareExchangePointer_rel:
    return MSVCIntrin::_InterlockedCompareExchange_rel;
  case clang::AArch64::BI_InterlockedCompareExchange8_nf:
  case clang::AArch64::BI_InterlockedCompareExchange16_nf:
  case clang::AArch64::BI_InterlockedCompareExchange_nf:
  case clang::AArch64::BI_InterlockedCompareExchange64_nf:
    return MSVCIntrin::_InterlockedCompareExchange_nf;
  case clang::AArch64::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::AArch64::BI_InterlockedCompareExchange128_acq:
    return MSVCIntrin::_InterlockedCompareExchange128_acq;
  case clang::AArch64::BI_InterlockedCompareExchange128_nf:
    return MSVCIntrin::_InterlockedCompareExchange128_nf;
  case clang::AArch64::BI_InterlockedCompareExchange128_rel:
    return MSVCIntrin::_InterlockedCompareExchange128_rel;
  case clang::AArch64::BI_InterlockedOr8_acq:
  case clang::AArch64::BI_InterlockedOr16_acq:
  case clang::AArch64::BI_InterlockedOr_acq:
  case clang::AArch64::BI_InterlockedOr64_acq:
    return MSVCIntrin::_InterlockedOr_acq;
  case clang::AArch64::BI_InterlockedOr8_rel:
  case clang::AArch64::BI_InterlockedOr16_rel:
  case clang::AArch64::BI_InterlockedOr_rel:
  case clang::AArch64::BI_InterlockedOr64_rel:
    return MSVCIntrin::_InterlockedOr_rel;
  case clang::AArch64::BI_InterlockedOr8_nf:
  case clang::AArch64::BI_InterlockedOr16_nf:
  case clang::AArch64::BI_InterlockedOr_nf:
  case clang::AArch64::BI_InterlockedOr64_nf:
    return MSVCIntrin::_InterlockedOr_nf;
  case clang::AArch64::BI_InterlockedXor8_acq:
  case clang::AArch64::BI_InterlockedXor16_acq:
  case clang::AArch64::BI_InterlockedXor_acq:
  case clang::AArch64::BI_InterlockedXor64_acq:
    return MSVCIntrin::_InterlockedXor_acq;
  case clang::AArch64::BI_InterlockedXor8_rel:
  case clang::AArch64::BI_InterlockedXor16_rel:
  case clang::AArch64::BI_InterlockedXor_rel:
  case clang::AArch64::BI_InterlockedXor64_rel:
    return MSVCIntrin::_InterlockedXor_rel;
  case clang::AArch64::BI_InterlockedXor8_nf:
  case clang::AArch64::BI_InterlockedXor16_nf:
  case clang::AArch64::BI_InterlockedXor_nf:
  case clang::AArch64::BI_InterlockedXor64_nf:
    return MSVCIntrin::_InterlockedXor_nf;
  case clang::AArch64::BI_InterlockedAnd8_acq:
  case clang::AArch64::BI_InterlockedAnd16_acq:
  case clang::AArch64::BI_InterlockedAnd_acq:
  case clang::AArch64::BI_InterlockedAnd64_acq:
    return MSVCIntrin::_InterlockedAnd_acq;
  case clang::AArch64::BI_InterlockedAnd8_rel:
  case clang::AArch64::BI_InterlockedAnd16_rel:
  case clang::AArch64::BI_InterlockedAnd_rel:
  case clang::AArch64::BI_InterlockedAnd64_rel:
    return MSVCIntrin::_InterlockedAnd_rel;
  case clang::AArch64::BI_InterlockedAnd8_nf:
  case clang::AArch64::BI_InterlockedAnd16_nf:
  case clang::AArch64::BI_InterlockedAnd_nf:
  case clang::AArch64::BI_InterlockedAnd64_nf:
    return MSVCIntrin::_InterlockedAnd_nf;
  case clang::AArch64::BI_InterlockedIncrement16_acq:
  case clang::AArch64::BI_InterlockedIncrement_acq:
  case clang::AArch64::BI_InterlockedIncrement64_acq:
    return MSVCIntrin::_InterlockedIncrement_acq;
  case clang::AArch64::BI_InterlockedIncrement16_rel:
  case clang::AArch64::BI_InterlockedIncrement_rel:
  case clang::AArch64::BI_InterlockedIncrement64_rel:
    return MSVCIntrin::_InterlockedIncrement_rel;
  case clang::AArch64::BI_InterlockedIncrement16_nf:
  case clang::AArch64::BI_InterlockedIncrement_nf:
  case clang::AArch64::BI_InterlockedIncrement64_nf:
    return MSVCIntrin::_InterlockedIncrement_nf;
  case clang::AArch64::BI_InterlockedDecrement16_acq:
  case clang::AArch64::BI_InterlockedDecrement_acq:
  case clang::AArch64::BI_InterlockedDecrement64_acq:
    return MSVCIntrin::_InterlockedDecrement_acq;
  case clang::AArch64::BI_InterlockedDecrement16_rel:
  case clang::AArch64::BI_InterlockedDecrement_rel:
  case clang::AArch64::BI_InterlockedDecrement64_rel:
    return MSVCIntrin::_InterlockedDecrement_rel;
  case clang::AArch64::BI_InterlockedDecrement16_nf:
  case clang::AArch64::BI_InterlockedDecrement_nf:
  case clang::AArch64::BI_InterlockedDecrement64_nf:
    return MSVCIntrin::_InterlockedDecrement_nf;
  }
  llvm_unreachable("must return from switch");
}

static std::optional<CodeGenFunction::MSVCIntrin>
translateArmToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CodeGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::ARM::BI_BitScanForward:
  case clang::ARM::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::ARM::BI_BitScanReverse:
  case clang::ARM::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::ARM::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::ARM::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::ARM::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::ARM::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::ARM::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::ARM::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::ARM::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::ARM::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  case clang::ARM::BI_InterlockedExchangeAdd8_acq:
  case clang::ARM::BI_InterlockedExchangeAdd16_acq:
  case clang::ARM::BI_InterlockedExchangeAdd_acq:
  case clang::ARM::BI_InterlockedExchangeAdd64_acq:
    return MSVCIntrin::_InterlockedExchangeAdd_acq;
  case clang::ARM::BI_InterlockedExchangeAdd8_rel:
  case clang::ARM::BI_InterlockedExchangeAdd16_rel:
  case clang::ARM::BI_InterlockedExchangeAdd_rel:
  case clang::ARM::BI_InterlockedExchangeAdd64_rel:
    return MSVCIntrin::_InterlockedExchangeAdd_rel;
  case clang::ARM::BI_InterlockedExchangeAdd8_nf:
  case clang::ARM::BI_InterlockedExchangeAdd16_nf:
  case clang::ARM::BI_InterlockedExchangeAdd_nf:
  case clang::ARM::BI_InterlockedExchangeAdd64_nf:
    return MSVCIntrin::_InterlockedExchangeAdd_nf;
  case clang::ARM::BI_InterlockedExchange8_acq:
  case clang::ARM::BI_InterlockedExchange16_acq:
  case clang::ARM::BI_InterlockedExchange_acq:
  case clang::ARM::BI_InterlockedExchange64_acq:
  case clang::ARM::BI_InterlockedExchangePointer_acq:
    return MSVCIntrin::_InterlockedExchange_acq;
  case clang::ARM::BI_InterlockedExchange8_rel:
  case clang::ARM::BI_InterlockedExchange16_rel:
  case clang::ARM::BI_InterlockedExchange_rel:
  case clang::ARM::BI_InterlockedExchange64_rel:
  case clang::ARM::BI_InterlockedExchangePointer_rel:
    return MSVCIntrin::_InterlockedExchange_rel;
  case clang::ARM::BI_InterlockedExchange8_nf:
  case clang::ARM::BI_InterlockedExchange16_nf:
  case clang::ARM::BI_InterlockedExchange_nf:
  case clang::ARM::BI_InterlockedExchange64_nf:
  case clang::ARM::BI_InterlockedExchangePointer_nf:
    return MSVCIntrin::_InterlockedExchange_nf;
  case clang::ARM::BI_InterlockedCompareExchange8_acq:
  case clang::ARM::BI_InterlockedCompareExchange16_acq:
  case clang::ARM::BI_InterlockedCompareExchange_acq:
  case clang::ARM::BI_InterlockedCompareExchange64_acq:
  case clang::ARM::BI_InterlockedCompareExchangePointer_acq:
    return MSVCIntrin::_InterlockedCompareExchange_acq;
  case clang::ARM::BI_InterlockedCompareExchange8_rel:
  case clang::ARM::BI_InterlockedCompareExchange16_rel:
  case clang::ARM::BI_InterlockedCompareExchange_rel:
  case clang::ARM::BI_InterlockedCompareExchange64_rel:
  case clang::ARM::BI_InterlockedCompareExchangePointer_rel:
    return MSVCIntrin::_InterlockedCompareExchange_rel;
  case clang::ARM::BI_InterlockedCompareExchange8_nf:
  case clang::ARM::BI_InterlockedCompareExchange16_nf:
  case clang::ARM::BI_InterlockedCompareExchange_nf:
  case clang::ARM::BI_InterlockedCompareExchange64_nf:
    return MSVCIntrin::_InterlockedCompareExchange_nf;
  case clang::ARM::BI_InterlockedOr8_acq:
  case clang::ARM::BI_InterlockedOr16_acq:
  case clang::ARM::BI_InterlockedOr_acq:
  case clang::ARM::BI_InterlockedOr64_acq:
    return MSVCIntrin::_InterlockedOr_acq;
  case clang::ARM::BI_InterlockedOr8_rel:
  case clang::ARM::BI_InterlockedOr16_rel:
  case clang::ARM::BI_InterlockedOr_rel:
  case clang::ARM::BI_InterlockedOr64_rel:
    return MSVCIntrin::_InterlockedOr_rel;
  case clang::ARM::BI_InterlockedOr8_nf:
  case clang::ARM::BI_InterlockedOr16_nf:
  case clang::ARM::BI_InterlockedOr_nf:
  case clang::ARM::BI_InterlockedOr64_nf:
    return MSVCIntrin::_InterlockedOr_nf;
  case clang::ARM::BI_InterlockedXor8_acq:
  case clang::ARM::BI_InterlockedXor16_acq:
  case clang::ARM::BI_InterlockedXor_acq:
  case clang::ARM::BI_InterlockedXor64_acq:
    return MSVCIntrin::_InterlockedXor_acq;
  case clang::ARM::BI_InterlockedXor8_rel:
  case clang::ARM::BI_InterlockedXor16_rel:
  case clang::ARM::BI_InterlockedXor_rel:
  case clang::ARM::BI_InterlockedXor64_rel:
    return MSVCIntrin::_InterlockedXor_rel;
  case clang::ARM::BI_InterlockedXor8_nf:
  case clang::ARM::BI_InterlockedXor16_nf:
  case clang::ARM::BI_InterlockedXor_nf:
  case clang::ARM::BI_InterlockedXor64_nf:
    return MSVCIntrin::_InterlockedXor_nf;
  case clang::ARM::BI_InterlockedAnd8_acq:
  case clang::ARM::BI_InterlockedAnd16_acq:
  case clang::ARM::BI_InterlockedAnd_acq:
  case clang::ARM::BI_InterlockedAnd64_acq:
    return MSVCIntrin::_InterlockedAnd_acq;
  case clang::ARM::BI_InterlockedAnd8_rel:
  case clang::ARM::BI_InterlockedAnd16_rel:
  case clang::ARM::BI_InterlockedAnd_rel:
  case clang::ARM::BI_InterlockedAnd64_rel:
    return MSVCIntrin::_InterlockedAnd_rel;
  case clang::ARM::BI_InterlockedAnd8_nf:
  case clang::ARM::BI_InterlockedAnd16_nf:
  case clang::ARM::BI_InterlockedAnd_nf:
  case clang::ARM::BI_InterlockedAnd64_nf:
    return MSVCIntrin::_InterlockedAnd_nf;
  case clang::ARM::BI_InterlockedIncrement16_acq:
  case clang::ARM::BI_InterlockedIncrement_acq:
  case clang::ARM::BI_InterlockedIncrement64_acq:
    return MSVCIntrin::_InterlockedIncrement_acq;
  case clang::ARM::BI_InterlockedIncrement16_rel:
  case clang::ARM::BI_InterlockedIncrement_rel:
  case clang::ARM::BI_InterlockedIncrement64_rel:
    return MSVCIntrin::_InterlockedIncrement_rel;
  case clang::ARM::BI_InterlockedIncrement16_nf:
  case clang::ARM::BI_InterlockedIncrement_nf:
  case clang::ARM::BI_InterlockedIncrement64_nf:
    return MSVCIntrin::_InterlockedIncrement_nf;
  case clang::ARM::BI_InterlockedDecrement16_acq:
  case clang::ARM::BI_InterlockedDecrement_acq:
  case clang::ARM::BI_InterlockedDecrement64_acq:
    return MSVCIntrin::_InterlockedDecrement_acq;
  case clang::ARM::BI_InterlockedDecrement16_rel:
  case clang::ARM::BI_InterlockedDecrement_rel:
  case clang::ARM::BI_InterlockedDecrement64_rel:
    return MSVCIntrin::_InterlockedDecrement_rel;
  case clang::ARM::BI_InterlockedDecrement16_nf:
  case clang::ARM::BI_InterlockedDecrement_nf:
  case clang::ARM::BI_InterlockedDecrement64_nf:
    return MSVCIntrin::_InterlockedDecrement_nf;
  }
  llvm_unreachable("must return from switch");
}

// Emit an intrinsic where all operands are of the same type as the result.
// Depending on mode, this may be a constrained floating-point intrinsic.
static Value *emitCallMaybeConstrainedFPBuiltin(CodeGenFunction &CGF,
                                                unsigned IntrinsicID,
                                                unsigned ConstrainedIntrinsicID,
                                                llvm::Type *Ty,
                                                ArrayRef<Value *> Args) {
  Function *F;
  if (CGF.Builder.getIsFPConstrained())
    F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID, Ty);
  else
    F = CGF.CGM.getIntrinsic(IntrinsicID, Ty);

  if (CGF.Builder.getIsFPConstrained())
    return CGF.Builder.CreateConstrainedFPCall(F, Args);
  else
    return CGF.Builder.CreateCall(F, Args);
}

static llvm::FixedVectorType *GetNeonType(CodeGenFunction *CGF,
                                          NeonTypeFlags TypeFlags,
                                          bool HasFastHalfType = true,
                                          bool V1Ty = false,
                                          bool AllowBFloatArgsAndRet = true) {
  int IsQuad = TypeFlags.isQuad();
  switch (TypeFlags.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
  case NeonTypeFlags::MFloat8:
    return llvm::FixedVectorType::get(CGF->Int8Ty, V1Ty ? 1 : (8 << IsQuad));
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
    return llvm::FixedVectorType::get(CGF->Int16Ty, V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::BFloat16:
    if (AllowBFloatArgsAndRet)
      return llvm::FixedVectorType::get(CGF->BFloatTy, V1Ty ? 1 : (4 << IsQuad));
    else
      return llvm::FixedVectorType::get(CGF->Int16Ty, V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::Float16:
    if (HasFastHalfType)
      return llvm::FixedVectorType::get(CGF->HalfTy, V1Ty ? 1 : (4 << IsQuad));
    else
      return llvm::FixedVectorType::get(CGF->Int16Ty, V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::Int32:
    return llvm::FixedVectorType::get(CGF->Int32Ty, V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Int64:
  case NeonTypeFlags::Poly64:
    return llvm::FixedVectorType::get(CGF->Int64Ty, V1Ty ? 1 : (1 << IsQuad));
  case NeonTypeFlags::Poly128:
    // FIXME: i128 and f128 doesn't get fully support in Clang and llvm.
    // There is a lot of i128 and f128 API missing.
    // so we use v16i8 to represent poly128 and get pattern matched.
    return llvm::FixedVectorType::get(CGF->Int8Ty, 16);
  case NeonTypeFlags::Float32:
    return llvm::FixedVectorType::get(CGF->FloatTy, V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Float64:
    return llvm::FixedVectorType::get(CGF->DoubleTy, V1Ty ? 1 : (1 << IsQuad));
  }
  llvm_unreachable("Unknown vector element type!");
}

static llvm::VectorType *GetFloatNeonType(CodeGenFunction *CGF,
                                          NeonTypeFlags IntTypeFlags) {
  int IsQuad = IntTypeFlags.isQuad();
  switch (IntTypeFlags.getEltType()) {
  case NeonTypeFlags::Int16:
    return llvm::FixedVectorType::get(CGF->HalfTy, (4 << IsQuad));
  case NeonTypeFlags::Int32:
    return llvm::FixedVectorType::get(CGF->FloatTy, (2 << IsQuad));
  case NeonTypeFlags::Int64:
    return llvm::FixedVectorType::get(CGF->DoubleTy, (1 << IsQuad));
  default:
    llvm_unreachable("Type can't be converted to floating-point!");
  }
}

Value *CodeGenFunction::EmitNeonSplat(Value *V, Constant *C,
                                      const ElementCount &Count) {
  Value *SV = llvm::ConstantVector::getSplat(Count, C);
  return Builder.CreateShuffleVector(V, V, SV, "lane");
}

Value *CodeGenFunction::EmitNeonSplat(Value *V, Constant *C) {
  ElementCount EC = cast<llvm::VectorType>(V->getType())->getElementCount();
  return EmitNeonSplat(V, C, EC);
}

Value *CodeGenFunction::EmitNeonCall(Function *F, SmallVectorImpl<Value*> &Ops,
                                     const char *name,
                                     unsigned shift, bool rightshift) {
  unsigned j = 0;
  for (Function::const_arg_iterator ai = F->arg_begin(), ae = F->arg_end();
       ai != ae; ++ai, ++j) {
    if (F->isConstrainedFPIntrinsic())
      if (ai->getType()->isMetadataTy())
        continue;
    if (shift > 0 && shift == j)
      Ops[j] = EmitNeonShiftVector(Ops[j], ai->getType(), rightshift);
    else
      Ops[j] = Builder.CreateBitCast(Ops[j], ai->getType(), name);
  }

  if (F->isConstrainedFPIntrinsic())
    return Builder.CreateConstrainedFPCall(F, Ops, name);
  else
    return Builder.CreateCall(F, Ops, name);
}

Value *CodeGenFunction::EmitFP8NeonCall(unsigned IID,
                                        ArrayRef<llvm::Type *> Tys,
                                        SmallVectorImpl<Value *> &Ops,
                                        const CallExpr *E, const char *name) {
  llvm::Value *FPM =
      EmitScalarOrConstFoldImmArg(/* ICEArguments */ 0, E->getNumArgs() - 1, E);
  Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_set_fpmr), FPM);
  return EmitNeonCall(CGM.getIntrinsic(IID, Tys), Ops, name);
}

llvm::Value *CodeGenFunction::EmitFP8NeonFDOTCall(
    unsigned IID, bool ExtendLaneArg, llvm::Type *RetTy,
    SmallVectorImpl<llvm::Value *> &Ops, const CallExpr *E, const char *name) {

  const unsigned ElemCount = Ops[0]->getType()->getPrimitiveSizeInBits() /
                             RetTy->getPrimitiveSizeInBits();
  llvm::Type *Tys[] = {llvm::FixedVectorType::get(RetTy, ElemCount),
                       Ops[1]->getType()};
  if (ExtendLaneArg) {
    auto *VT = llvm::FixedVectorType::get(Int8Ty, 16);
    Ops[2] = Builder.CreateInsertVector(VT, PoisonValue::get(VT), Ops[2],
                                        uint64_t(0));
  }
  return EmitFP8NeonCall(IID, Tys, Ops, E, name);
}

llvm::Value *CodeGenFunction::EmitFP8NeonFMLACall(
    unsigned IID, bool ExtendLaneArg, llvm::Type *RetTy,
    SmallVectorImpl<llvm::Value *> &Ops, const CallExpr *E, const char *name) {

  if (ExtendLaneArg) {
    auto *VT = llvm::FixedVectorType::get(Int8Ty, 16);
    Ops[2] = Builder.CreateInsertVector(VT, PoisonValue::get(VT), Ops[2],
                                        uint64_t(0));
  }
  const unsigned ElemCount = Ops[0]->getType()->getPrimitiveSizeInBits() /
                             RetTy->getPrimitiveSizeInBits();
  return EmitFP8NeonCall(IID, {llvm::FixedVectorType::get(RetTy, ElemCount)},
                         Ops, E, name);
}

Value *CodeGenFunction::EmitNeonShiftVector(Value *V, llvm::Type *Ty,
                                            bool neg) {
  int SV = cast<ConstantInt>(V)->getSExtValue();
  return ConstantInt::get(Ty, neg ? -SV : SV);
}

Value *CodeGenFunction::EmitFP8NeonCvtCall(unsigned IID, llvm::Type *Ty0,
                                           llvm::Type *Ty1, bool Extract,
                                           SmallVectorImpl<llvm::Value *> &Ops,
                                           const CallExpr *E,
                                           const char *name) {
  llvm::Type *Tys[] = {Ty0, Ty1};
  if (Extract) {
    // Op[0] is mfloat8x16_t, but the intrinsic converts only the lower part of
    // the vector.
    Tys[1] = llvm::FixedVectorType::get(Int8Ty, 8);
    Ops[0] = Builder.CreateExtractVector(Tys[1], Ops[0], uint64_t(0));
  }
  return EmitFP8NeonCall(IID, Tys, Ops, E, name);
}

// Right-shift a vector by a constant.
Value *CodeGenFunction::EmitNeonRShiftImm(Value *Vec, Value *Shift,
                                          llvm::Type *Ty, bool usgn,
                                          const char *name) {
  llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);

  int ShiftAmt = cast<ConstantInt>(Shift)->getSExtValue();
  int EltSize = VTy->getScalarSizeInBits();

  Vec = Builder.CreateBitCast(Vec, Ty);

  // lshr/ashr are undefined when the shift amount is equal to the vector
  // element size.
  if (ShiftAmt == EltSize) {
    if (usgn) {
      // Right-shifting an unsigned value by its size yields 0.
      return llvm::ConstantAggregateZero::get(VTy);
    } else {
      // Right-shifting a signed value by its size is equivalent
      // to a shift of size-1.
      --ShiftAmt;
      Shift = ConstantInt::get(VTy->getElementType(), ShiftAmt);
    }
  }

  Shift = EmitNeonShiftVector(Shift, Ty, false);
  if (usgn)
    return Builder.CreateLShr(Vec, Shift, name);
  else
    return Builder.CreateAShr(Vec, Shift, name);
}

enum {
  AddRetType = (1 << 0),
  Add1ArgType = (1 << 1),
  Add2ArgTypes = (1 << 2),

  VectorizeRetType = (1 << 3),
  VectorizeArgTypes = (1 << 4),

  InventFloatType = (1 << 5),
  UnsignedAlts = (1 << 6),

  Use64BitVectors = (1 << 7),
  Use128BitVectors = (1 << 8),

  Vectorize1ArgType = Add1ArgType | VectorizeArgTypes,
  VectorRet = AddRetType | VectorizeRetType,
  VectorRetGetArgs01 =
      AddRetType | Add2ArgTypes | VectorizeRetType | VectorizeArgTypes,
  FpCmpzModifiers =
      AddRetType | VectorizeRetType | Add1ArgType | InventFloatType
};

namespace {
struct ARMVectorIntrinsicInfo {
  const char *NameHint;
  unsigned BuiltinID;
  unsigned LLVMIntrinsic;
  unsigned AltLLVMIntrinsic;
  uint64_t TypeModifier;

  bool operator<(unsigned RHSBuiltinID) const {
    return BuiltinID < RHSBuiltinID;
  }
  bool operator<(const ARMVectorIntrinsicInfo &TE) const {
    return BuiltinID < TE.BuiltinID;
  }
};
} // end anonymous namespace

#define NEONMAP0(NameBase) \
  { #NameBase, NEON::BI__builtin_neon_ ## NameBase, 0, 0, 0 }

#define NEONMAP1(NameBase, LLVMIntrinsic, TypeModifier) \
  { #NameBase, NEON:: BI__builtin_neon_ ## NameBase, \
      Intrinsic::LLVMIntrinsic, 0, TypeModifier }

#define NEONMAP2(NameBase, LLVMIntrinsic, AltLLVMIntrinsic, TypeModifier) \
  { #NameBase, NEON:: BI__builtin_neon_ ## NameBase, \
      Intrinsic::LLVMIntrinsic, Intrinsic::AltLLVMIntrinsic, \
      TypeModifier }

static const ARMVectorIntrinsicInfo ARMSIMDIntrinsicMap [] = {
  NEONMAP1(__a32_vcvt_bf16_f32, arm_neon_vcvtfp2bf, 0),
  NEONMAP0(splat_lane_v),
  NEONMAP0(splat_laneq_v),
  NEONMAP0(splatq_lane_v),
  NEONMAP0(splatq_laneq_v),
  NEONMAP2(vabd_v, arm_neon_vabdu, arm_neon_vabds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vabdq_v, arm_neon_vabdu, arm_neon_vabds, Add1ArgType | UnsignedAlts),
  NEONMAP1(vabs_v, arm_neon_vabs, 0),
  NEONMAP1(vabsq_v, arm_neon_vabs, 0),
  NEONMAP0(vadd_v),
  NEONMAP0(vaddhn_v),
  NEONMAP0(vaddq_v),
  NEONMAP1(vaesdq_u8, arm_neon_aesd, 0),
  NEONMAP1(vaeseq_u8, arm_neon_aese, 0),
  NEONMAP1(vaesimcq_u8, arm_neon_aesimc, 0),
  NEONMAP1(vaesmcq_u8, arm_neon_aesmc, 0),
  NEONMAP1(vbfdot_f32, arm_neon_bfdot, 0),
  NEONMAP1(vbfdotq_f32, arm_neon_bfdot, 0),
  NEONMAP1(vbfmlalbq_f32, arm_neon_bfmlalb, 0),
  NEONMAP1(vbfmlaltq_f32, arm_neon_bfmlalt, 0),
  NEONMAP1(vbfmmlaq_f32, arm_neon_bfmmla, 0),
  NEONMAP1(vbsl_v, arm_neon_vbsl, AddRetType),
  NEONMAP1(vbslq_v, arm_neon_vbsl, AddRetType),
  NEONMAP1(vcadd_rot270_f16, arm_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcadd_rot270_f32, arm_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcadd_rot90_f16, arm_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcadd_rot90_f32, arm_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f16, arm_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f32, arm_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f64, arm_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f16, arm_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f32, arm_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f64, arm_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcage_v, arm_neon_vacge, 0),
  NEONMAP1(vcageq_v, arm_neon_vacge, 0),
  NEONMAP1(vcagt_v, arm_neon_vacgt, 0),
  NEONMAP1(vcagtq_v, arm_neon_vacgt, 0),
  NEONMAP1(vcale_v, arm_neon_vacge, 0),
  NEONMAP1(vcaleq_v, arm_neon_vacge, 0),
  NEONMAP1(vcalt_v, arm_neon_vacgt, 0),
  NEONMAP1(vcaltq_v, arm_neon_vacgt, 0),
  NEONMAP0(vceqz_v),
  NEONMAP0(vceqzq_v),
  NEONMAP0(vcgez_v),
  NEONMAP0(vcgezq_v),
  NEONMAP0(vcgtz_v),
  NEONMAP0(vcgtzq_v),
  NEONMAP0(vclez_v),
  NEONMAP0(vclezq_v),
  NEONMAP1(vcls_v, arm_neon_vcls, Add1ArgType),
  NEONMAP1(vclsq_v, arm_neon_vcls, Add1ArgType),
  NEONMAP0(vcltz_v),
  NEONMAP0(vcltzq_v),
  NEONMAP1(vclz_v, ctlz, Add1ArgType),
  NEONMAP1(vclzq_v, ctlz, Add1ArgType),
  NEONMAP1(vcnt_v, ctpop, Add1ArgType),
  NEONMAP1(vcntq_v, ctpop, Add1ArgType),
  NEONMAP1(vcvt_f16_f32, arm_neon_vcvtfp2hf, 0),
  NEONMAP0(vcvt_f16_s16),
  NEONMAP0(vcvt_f16_u16),
  NEONMAP1(vcvt_f32_f16, arm_neon_vcvthf2fp, 0),
  NEONMAP0(vcvt_f32_v),
  NEONMAP1(vcvt_n_f16_s16, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvt_n_f16_u16, arm_neon_vcvtfxu2fp, 0),
  NEONMAP2(vcvt_n_f32_v, arm_neon_vcvtfxu2fp, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvt_n_s16_f16, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_s32_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_s64_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_u16_f16, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvt_n_u32_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvt_n_u64_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP0(vcvt_s16_f16),
  NEONMAP0(vcvt_s32_v),
  NEONMAP0(vcvt_s64_v),
  NEONMAP0(vcvt_u16_f16),
  NEONMAP0(vcvt_u32_v),
  NEONMAP0(vcvt_u64_v),
  NEONMAP1(vcvta_s16_f16, arm_neon_vcvtas, 0),
  NEONMAP1(vcvta_s32_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvta_s64_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvta_u16_f16, arm_neon_vcvtau, 0),
  NEONMAP1(vcvta_u32_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvta_u64_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtaq_s16_f16, arm_neon_vcvtas, 0),
  NEONMAP1(vcvtaq_s32_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvtaq_s64_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvtaq_u16_f16, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtaq_u32_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtaq_u64_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvth_bf16_f32, arm_neon_vcvtbfp2bf, 0),
  NEONMAP1(vcvtm_s16_f16, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtm_s32_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtm_s64_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtm_u16_f16, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtm_u32_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtm_u64_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtmq_s16_f16, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtmq_s32_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtmq_s64_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtmq_u16_f16, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtmq_u32_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtmq_u64_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtn_s16_f16, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtn_s32_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtn_s64_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtn_u16_f16, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtn_u32_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtn_u64_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtnq_s16_f16, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtnq_s32_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtnq_s64_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtnq_u16_f16, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtnq_u32_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtnq_u64_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtp_s16_f16, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtp_s32_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtp_s64_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtp_u16_f16, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtp_u32_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtp_u64_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtpq_s16_f16, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtpq_s32_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtpq_s64_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtpq_u16_f16, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtpq_u32_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtpq_u64_v, arm_neon_vcvtpu, 0),
  NEONMAP0(vcvtq_f16_s16),
  NEONMAP0(vcvtq_f16_u16),
  NEONMAP0(vcvtq_f32_v),
  NEONMAP1(vcvtq_n_f16_s16, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvtq_n_f16_u16, arm_neon_vcvtfxu2fp, 0),
  NEONMAP2(vcvtq_n_f32_v, arm_neon_vcvtfxu2fp, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvtq_n_s16_f16, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_s32_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_s64_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_u16_f16, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtq_n_u32_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtq_n_u64_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP0(vcvtq_s16_f16),
  NEONMAP0(vcvtq_s32_v),
  NEONMAP0(vcvtq_s64_v),
  NEONMAP0(vcvtq_u16_f16),
  NEONMAP0(vcvtq_u32_v),
  NEONMAP0(vcvtq_u64_v),
  NEONMAP1(vdot_s32, arm_neon_sdot, 0),
  NEONMAP1(vdot_u32, arm_neon_udot, 0),
  NEONMAP1(vdotq_s32, arm_neon_sdot, 0),
  NEONMAP1(vdotq_u32, arm_neon_udot, 0),
  NEONMAP0(vext_v),
  NEONMAP0(vextq_v),
  NEONMAP0(vfma_v),
  NEONMAP0(vfmaq_v),
  NEONMAP2(vhadd_v, arm_neon_vhaddu, arm_neon_vhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhaddq_v, arm_neon_vhaddu, arm_neon_vhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsub_v, arm_neon_vhsubu, arm_neon_vhsubs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsubq_v, arm_neon_vhsubu, arm_neon_vhsubs, Add1ArgType | UnsignedAlts),
  NEONMAP0(vld1_dup_v),
  NEONMAP1(vld1_v, arm_neon_vld1, 0),
  NEONMAP1(vld1_x2_v, arm_neon_vld1x2, 0),
  NEONMAP1(vld1_x3_v, arm_neon_vld1x3, 0),
  NEONMAP1(vld1_x4_v, arm_neon_vld1x4, 0),
  NEONMAP0(vld1q_dup_v),
  NEONMAP1(vld1q_v, arm_neon_vld1, 0),
  NEONMAP1(vld1q_x2_v, arm_neon_vld1x2, 0),
  NEONMAP1(vld1q_x3_v, arm_neon_vld1x3, 0),
  NEONMAP1(vld1q_x4_v, arm_neon_vld1x4, 0),
  NEONMAP1(vld2_dup_v, arm_neon_vld2dup, 0),
  NEONMAP1(vld2_lane_v, arm_neon_vld2lane, 0),
  NEONMAP1(vld2_v, arm_neon_vld2, 0),
  NEONMAP1(vld2q_dup_v, arm_neon_vld2dup, 0),
  NEONMAP1(vld2q_lane_v, arm_neon_vld2lane, 0),
  NEONMAP1(vld2q_v, arm_neon_vld2, 0),
  NEONMAP1(vld3_dup_v, arm_neon_vld3dup, 0),
  NEONMAP1(vld3_lane_v, arm_neon_vld3lane, 0),
  NEONMAP1(vld3_v, arm_neon_vld3, 0),
  NEONMAP1(vld3q_dup_v, arm_neon_vld3dup, 0),
  NEONMAP1(vld3q_lane_v, arm_neon_vld3lane, 0),
  NEONMAP1(vld3q_v, arm_neon_vld3, 0),
  NEONMAP1(vld4_dup_v, arm_neon_vld4dup, 0),
  NEONMAP1(vld4_lane_v, arm_neon_vld4lane, 0),
  NEONMAP1(vld4_v, arm_neon_vld4, 0),
  NEONMAP1(vld4q_dup_v, arm_neon_vld4dup, 0),
  NEONMAP1(vld4q_lane_v, arm_neon_vld4lane, 0),
  NEONMAP1(vld4q_v, arm_neon_vld4, 0),
  NEONMAP2(vmax_v, arm_neon_vmaxu, arm_neon_vmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP1(vmaxnm_v, arm_neon_vmaxnm, Add1ArgType),
  NEONMAP1(vmaxnmq_v, arm_neon_vmaxnm, Add1ArgType),
  NEONMAP2(vmaxq_v, arm_neon_vmaxu, arm_neon_vmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vmin_v, arm_neon_vminu, arm_neon_vmins, Add1ArgType | UnsignedAlts),
  NEONMAP1(vminnm_v, arm_neon_vminnm, Add1ArgType),
  NEONMAP1(vminnmq_v, arm_neon_vminnm, Add1ArgType),
  NEONMAP2(vminq_v, arm_neon_vminu, arm_neon_vmins, Add1ArgType | UnsignedAlts),
  NEONMAP1(vmmlaq_s32, arm_neon_smmla, 0),
  NEONMAP1(vmmlaq_u32, arm_neon_ummla, 0),
  NEONMAP0(vmovl_v),
  NEONMAP0(vmovn_v),
  NEONMAP1(vmul_v, arm_neon_vmulp, Add1ArgType),
  NEONMAP0(vmull_v),
  NEONMAP1(vmulq_v, arm_neon_vmulp, Add1ArgType),
  NEONMAP2(vpadal_v, arm_neon_vpadalu, arm_neon_vpadals, UnsignedAlts),
  NEONMAP2(vpadalq_v, arm_neon_vpadalu, arm_neon_vpadals, UnsignedAlts),
  NEONMAP1(vpadd_v, arm_neon_vpadd, Add1ArgType),
  NEONMAP2(vpaddl_v, arm_neon_vpaddlu, arm_neon_vpaddls, UnsignedAlts),
  NEONMAP2(vpaddlq_v, arm_neon_vpaddlu, arm_neon_vpaddls, UnsignedAlts),
  NEONMAP1(vpaddq_v, arm_neon_vpadd, Add1ArgType),
  NEONMAP2(vpmax_v, arm_neon_vpmaxu, arm_neon_vpmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vpmin_v, arm_neon_vpminu, arm_neon_vpmins, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqabs_v, arm_neon_vqabs, Add1ArgType),
  NEONMAP1(vqabsq_v, arm_neon_vqabs, Add1ArgType),
  NEONMAP2(vqadd_v, uadd_sat, sadd_sat, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqaddq_v, uadd_sat, sadd_sat, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqdmlal_v, arm_neon_vqdmull, sadd_sat, 0),
  NEONMAP2(vqdmlsl_v, arm_neon_vqdmull, ssub_sat, 0),
  NEONMAP1(vqdmulh_v, arm_neon_vqdmulh, Add1ArgType),
  NEONMAP1(vqdmulhq_v, arm_neon_vqdmulh, Add1ArgType),
  NEONMAP1(vqdmull_v, arm_neon_vqdmull, Add1ArgType),
  NEONMAP2(vqmovn_v, arm_neon_vqmovnu, arm_neon_vqmovns, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqmovun_v, arm_neon_vqmovnsu, Add1ArgType),
  NEONMAP1(vqneg_v, arm_neon_vqneg, Add1ArgType),
  NEONMAP1(vqnegq_v, arm_neon_vqneg, Add1ArgType),
  NEONMAP1(vqrdmlah_s16, arm_neon_vqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlah_s32, arm_neon_vqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlahq_s16, arm_neon_vqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlahq_s32, arm_neon_vqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlsh_s16, arm_neon_vqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlsh_s32, arm_neon_vqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlshq_s16, arm_neon_vqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlshq_s32, arm_neon_vqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmulh_v, arm_neon_vqrdmulh, Add1ArgType),
  NEONMAP1(vqrdmulhq_v, arm_neon_vqrdmulh, Add1ArgType),
  NEONMAP2(vqrshl_v, arm_neon_vqrshiftu, arm_neon_vqrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqrshlq_v, arm_neon_vqrshiftu, arm_neon_vqrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshl_n_v, arm_neon_vqshiftu, arm_neon_vqshifts, UnsignedAlts),
  NEONMAP2(vqshl_v, arm_neon_vqshiftu, arm_neon_vqshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshlq_n_v, arm_neon_vqshiftu, arm_neon_vqshifts, UnsignedAlts),
  NEONMAP2(vqshlq_v, arm_neon_vqshiftu, arm_neon_vqshifts, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqshlu_n_v, arm_neon_vqshiftsu, 0),
  NEONMAP1(vqshluq_n_v, arm_neon_vqshiftsu, 0),
  NEONMAP2(vqsub_v, usub_sat, ssub_sat, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqsubq_v, usub_sat, ssub_sat, Add1ArgType | UnsignedAlts),
  NEONMAP1(vraddhn_v, arm_neon_vraddhn, Add1ArgType),
  NEONMAP2(vrecpe_v, arm_neon_vrecpe, arm_neon_vrecpe, 0),
  NEONMAP2(vrecpeq_v, arm_neon_vrecpe, arm_neon_vrecpe, 0),
  NEONMAP1(vrecps_v, arm_neon_vrecps, Add1ArgType),
  NEONMAP1(vrecpsq_v, arm_neon_vrecps, Add1ArgType),
  NEONMAP2(vrhadd_v, arm_neon_vrhaddu, arm_neon_vrhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrhaddq_v, arm_neon_vrhaddu, arm_neon_vrhadds, Add1ArgType | UnsignedAlts),
  NEONMAP1(vrnd_v, trunc, Add1ArgType),
  NEONMAP1(vrnda_v, round, Add1ArgType),
  NEONMAP1(vrndaq_v, round, Add1ArgType),
  NEONMAP0(vrndi_v),
  NEONMAP0(vrndiq_v),
  NEONMAP1(vrndm_v, floor, Add1ArgType),
  NEONMAP1(vrndmq_v, floor, Add1ArgType),
  NEONMAP1(vrndn_v, roundeven, Add1ArgType),
  NEONMAP1(vrndnq_v, roundeven, Add1ArgType),
  NEONMAP1(vrndp_v, ceil, Add1ArgType),
  NEONMAP1(vrndpq_v, ceil, Add1ArgType),
  NEONMAP1(vrndq_v, trunc, Add1ArgType),
  NEONMAP1(vrndx_v, rint, Add1ArgType),
  NEONMAP1(vrndxq_v, rint, Add1ArgType),
  NEONMAP2(vrshl_v, arm_neon_vrshiftu, arm_neon_vrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshlq_v, arm_neon_vrshiftu, arm_neon_vrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshr_n_v, arm_neon_vrshiftu, arm_neon_vrshifts, UnsignedAlts),
  NEONMAP2(vrshrq_n_v, arm_neon_vrshiftu, arm_neon_vrshifts, UnsignedAlts),
  NEONMAP2(vrsqrte_v, arm_neon_vrsqrte, arm_neon_vrsqrte, 0),
  NEONMAP2(vrsqrteq_v, arm_neon_vrsqrte, arm_neon_vrsqrte, 0),
  NEONMAP1(vrsqrts_v, arm_neon_vrsqrts, Add1ArgType),
  NEONMAP1(vrsqrtsq_v, arm_neon_vrsqrts, Add1ArgType),
  NEONMAP1(vrsubhn_v, arm_neon_vrsubhn, Add1ArgType),
  NEONMAP1(vsha1su0q_u32, arm_neon_sha1su0, 0),
  NEONMAP1(vsha1su1q_u32, arm_neon_sha1su1, 0),
  NEONMAP1(vsha256h2q_u32, arm_neon_sha256h2, 0),
  NEONMAP1(vsha256hq_u32, arm_neon_sha256h, 0),
  NEONMAP1(vsha256su0q_u32, arm_neon_sha256su0, 0),
  NEONMAP1(vsha256su1q_u32, arm_neon_sha256su1, 0),
  NEONMAP0(vshl_n_v),
  NEONMAP2(vshl_v, arm_neon_vshiftu, arm_neon_vshifts, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshll_n_v),
  NEONMAP0(vshlq_n_v),
  NEONMAP2(vshlq_v, arm_neon_vshiftu, arm_neon_vshifts, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshr_n_v),
  NEONMAP0(vshrn_n_v),
  NEONMAP0(vshrq_n_v),
  NEONMAP1(vst1_v, arm_neon_vst1, 0),
  NEONMAP1(vst1_x2_v, arm_neon_vst1x2, 0),
  NEONMAP1(vst1_x3_v, arm_neon_vst1x3, 0),
  NEONMAP1(vst1_x4_v, arm_neon_vst1x4, 0),
  NEONMAP1(vst1q_v, arm_neon_vst1, 0),
  NEONMAP1(vst1q_x2_v, arm_neon_vst1x2, 0),
  NEONMAP1(vst1q_x3_v, arm_neon_vst1x3, 0),
  NEONMAP1(vst1q_x4_v, arm_neon_vst1x4, 0),
  NEONMAP1(vst2_lane_v, arm_neon_vst2lane, 0),
  NEONMAP1(vst2_v, arm_neon_vst2, 0),
  NEONMAP1(vst2q_lane_v, arm_neon_vst2lane, 0),
  NEONMAP1(vst2q_v, arm_neon_vst2, 0),
  NEONMAP1(vst3_lane_v, arm_neon_vst3lane, 0),
  NEONMAP1(vst3_v, arm_neon_vst3, 0),
  NEONMAP1(vst3q_lane_v, arm_neon_vst3lane, 0),
  NEONMAP1(vst3q_v, arm_neon_vst3, 0),
  NEONMAP1(vst4_lane_v, arm_neon_vst4lane, 0),
  NEONMAP1(vst4_v, arm_neon_vst4, 0),
  NEONMAP1(vst4q_lane_v, arm_neon_vst4lane, 0),
  NEONMAP1(vst4q_v, arm_neon_vst4, 0),
  NEONMAP0(vsubhn_v),
  NEONMAP0(vtrn_v),
  NEONMAP0(vtrnq_v),
  NEONMAP0(vtst_v),
  NEONMAP0(vtstq_v),
  NEONMAP1(vusdot_s32, arm_neon_usdot, 0),
  NEONMAP1(vusdotq_s32, arm_neon_usdot, 0),
  NEONMAP1(vusmmlaq_s32, arm_neon_usmmla, 0),
  NEONMAP0(vuzp_v),
  NEONMAP0(vuzpq_v),
  NEONMAP0(vzip_v),
  NEONMAP0(vzipq_v)
};

static const ARMVectorIntrinsicInfo AArch64SIMDIntrinsicMap[] = {
  NEONMAP0(splat_lane_v),
  NEONMAP0(splat_laneq_v),
  NEONMAP0(splatq_lane_v),
  NEONMAP0(splatq_laneq_v),
  NEONMAP1(vabs_v, aarch64_neon_abs, 0),
  NEONMAP1(vabsq_v, aarch64_neon_abs, 0),
  NEONMAP0(vadd_v),
  NEONMAP0(vaddhn_v),
  NEONMAP0(vaddq_p128),
  NEONMAP0(vaddq_v),
  NEONMAP1(vaesdq_u8, aarch64_crypto_aesd, 0),
  NEONMAP1(vaeseq_u8, aarch64_crypto_aese, 0),
  NEONMAP1(vaesimcq_u8, aarch64_crypto_aesimc, 0),
  NEONMAP1(vaesmcq_u8, aarch64_crypto_aesmc, 0),
  NEONMAP2(vbcaxq_s16, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_s32, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_s64, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_s8, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_u16, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_u32, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_u64, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vbcaxq_u8, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs, Add1ArgType | UnsignedAlts),
  NEONMAP1(vbfdot_f32, aarch64_neon_bfdot, 0),
  NEONMAP1(vbfdotq_f32, aarch64_neon_bfdot, 0),
  NEONMAP1(vbfmlalbq_f32, aarch64_neon_bfmlalb, 0),
  NEONMAP1(vbfmlaltq_f32, aarch64_neon_bfmlalt, 0),
  NEONMAP1(vbfmmlaq_f32, aarch64_neon_bfmmla, 0),
  NEONMAP1(vcadd_rot270_f16, aarch64_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcadd_rot270_f32, aarch64_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcadd_rot90_f16, aarch64_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcadd_rot90_f32, aarch64_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f16, aarch64_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f32, aarch64_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot270_f64, aarch64_neon_vcadd_rot270, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f16, aarch64_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f32, aarch64_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcaddq_rot90_f64, aarch64_neon_vcadd_rot90, Add1ArgType),
  NEONMAP1(vcage_v, aarch64_neon_facge, 0),
  NEONMAP1(vcageq_v, aarch64_neon_facge, 0),
  NEONMAP1(vcagt_v, aarch64_neon_facgt, 0),
  NEONMAP1(vcagtq_v, aarch64_neon_facgt, 0),
  NEONMAP1(vcale_v, aarch64_neon_facge, 0),
  NEONMAP1(vcaleq_v, aarch64_neon_facge, 0),
  NEONMAP1(vcalt_v, aarch64_neon_facgt, 0),
  NEONMAP1(vcaltq_v, aarch64_neon_facgt, 0),
  NEONMAP0(vceqz_v),
  NEONMAP0(vceqzq_v),
  NEONMAP0(vcgez_v),
  NEONMAP0(vcgezq_v),
  NEONMAP0(vcgtz_v),
  NEONMAP0(vcgtzq_v),
  NEONMAP0(vclez_v),
  NEONMAP0(vclezq_v),
  NEONMAP1(vcls_v, aarch64_neon_cls, Add1ArgType),
  NEONMAP1(vclsq_v, aarch64_neon_cls, Add1ArgType),
  NEONMAP0(vcltz_v),
  NEONMAP0(vcltzq_v),
  NEONMAP1(vclz_v, ctlz, Add1ArgType),
  NEONMAP1(vclzq_v, ctlz, Add1ArgType),
  NEONMAP1(vcmla_f16, aarch64_neon_vcmla_rot0, Add1ArgType),
  NEONMAP1(vcmla_f32, aarch64_neon_vcmla_rot0, Add1ArgType),
  NEONMAP1(vcmla_rot180_f16, aarch64_neon_vcmla_rot180, Add1ArgType),
  NEONMAP1(vcmla_rot180_f32, aarch64_neon_vcmla_rot180, Add1ArgType),
  NEONMAP1(vcmla_rot270_f16, aarch64_neon_vcmla_rot270, Add1ArgType),
  NEONMAP1(vcmla_rot270_f32, aarch64_neon_vcmla_rot270, Add1ArgType),
  NEONMAP1(vcmla_rot90_f16, aarch64_neon_vcmla_rot90, Add1ArgType),
  NEONMAP1(vcmla_rot90_f32, aarch64_neon_vcmla_rot90, Add1ArgType),
  NEONMAP1(vcmlaq_f16, aarch64_neon_vcmla_rot0, Add1ArgType),
  NEONMAP1(vcmlaq_f32, aarch64_neon_vcmla_rot0, Add1ArgType),
  NEONMAP1(vcmlaq_f64, aarch64_neon_vcmla_rot0, Add1ArgType),
  NEONMAP1(vcmlaq_rot180_f16, aarch64_neon_vcmla_rot180, Add1ArgType),
  NEONMAP1(vcmlaq_rot180_f32, aarch64_neon_vcmla_rot180, Add1ArgType),
  NEONMAP1(vcmlaq_rot180_f64, aarch64_neon_vcmla_rot180, Add1ArgType),
  NEONMAP1(vcmlaq_rot270_f16, aarch64_neon_vcmla_rot270, Add1ArgType),
  NEONMAP1(vcmlaq_rot270_f32, aarch64_neon_vcmla_rot270, Add1ArgType),
  NEONMAP1(vcmlaq_rot270_f64, aarch64_neon_vcmla_rot270, Add1ArgType),
  NEONMAP1(vcmlaq_rot90_f16, aarch64_neon_vcmla_rot90, Add1ArgType),
  NEONMAP1(vcmlaq_rot90_f32, aarch64_neon_vcmla_rot90, Add1ArgType),
  NEONMAP1(vcmlaq_rot90_f64, aarch64_neon_vcmla_rot90, Add1ArgType),
  NEONMAP1(vcnt_v, ctpop, Add1ArgType),
  NEONMAP1(vcntq_v, ctpop, Add1ArgType),
  NEONMAP1(vcvt_f16_f32, aarch64_neon_vcvtfp2hf, 0),
  NEONMAP0(vcvt_f16_s16),
  NEONMAP0(vcvt_f16_u16),
  NEONMAP1(vcvt_f32_f16, aarch64_neon_vcvthf2fp, 0),
  NEONMAP0(vcvt_f32_v),
  NEONMAP1(vcvt_n_f16_s16, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvt_n_f16_u16, aarch64_neon_vcvtfxu2fp, 0),
  NEONMAP2(vcvt_n_f32_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP2(vcvt_n_f64_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvt_n_s16_f16, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_s32_v, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_s64_v, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_u16_f16, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvt_n_u32_v, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvt_n_u64_v, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP0(vcvtq_f16_s16),
  NEONMAP0(vcvtq_f16_u16),
  NEONMAP0(vcvtq_f32_v),
  NEONMAP0(vcvtq_high_bf16_f32),
  NEONMAP0(vcvtq_low_bf16_f32),
  NEONMAP1(vcvtq_n_f16_s16, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvtq_n_f16_u16, aarch64_neon_vcvtfxu2fp, 0),
  NEONMAP2(vcvtq_n_f32_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP2(vcvtq_n_f64_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvtq_n_s16_f16, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_s32_v, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_s64_v, aarch64_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_u16_f16, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtq_n_u32_v, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtq_n_u64_v, aarch64_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtx_f32_v, aarch64_neon_fcvtxn, AddRetType | Add1ArgType),
  NEONMAP1(vdot_s32, aarch64_neon_sdot, 0),
  NEONMAP1(vdot_u32, aarch64_neon_udot, 0),
  NEONMAP1(vdotq_s32, aarch64_neon_sdot, 0),
  NEONMAP1(vdotq_u32, aarch64_neon_udot, 0),
  NEONMAP2(veor3q_s16, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_s32, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_s64, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_s8, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_u16, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_u32, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_u64, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP2(veor3q_u8, aarch64_crypto_eor3u, aarch64_crypto_eor3s, Add1ArgType | UnsignedAlts),
  NEONMAP0(vext_v),
  NEONMAP0(vextq_v),
  NEONMAP0(vfma_v),
  NEONMAP0(vfmaq_v),
  NEONMAP1(vfmlal_high_f16, aarch64_neon_fmlal2, 0),
  NEONMAP1(vfmlal_low_f16, aarch64_neon_fmlal, 0),
  NEONMAP1(vfmlalq_high_f16, aarch64_neon_fmlal2, 0),
  NEONMAP1(vfmlalq_low_f16, aarch64_neon_fmlal, 0),
  NEONMAP1(vfmlsl_high_f16, aarch64_neon_fmlsl2, 0),
  NEONMAP1(vfmlsl_low_f16, aarch64_neon_fmlsl, 0),
  NEONMAP1(vfmlslq_high_f16, aarch64_neon_fmlsl2, 0),
  NEONMAP1(vfmlslq_low_f16, aarch64_neon_fmlsl, 0),
  NEONMAP2(vhadd_v, aarch64_neon_uhadd, aarch64_neon_shadd, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhaddq_v, aarch64_neon_uhadd, aarch64_neon_shadd, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsub_v, aarch64_neon_uhsub, aarch64_neon_shsub, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsubq_v, aarch64_neon_uhsub, aarch64_neon_shsub, Add1ArgType | UnsignedAlts),
  NEONMAP1(vld1_x2_v, aarch64_neon_ld1x2, 0),
  NEONMAP1(vld1_x3_v, aarch64_neon_ld1x3, 0),
  NEONMAP1(vld1_x4_v, aarch64_neon_ld1x4, 0),
  NEONMAP1(vld1q_x2_v, aarch64_neon_ld1x2, 0),
  NEONMAP1(vld1q_x3_v, aarch64_neon_ld1x3, 0),
  NEONMAP1(vld1q_x4_v, aarch64_neon_ld1x4, 0),
  NEONMAP1(vmmlaq_s32, aarch64_neon_smmla, 0),
  NEONMAP1(vmmlaq_u32, aarch64_neon_ummla, 0),
  NEONMAP0(vmovl_v),
  NEONMAP0(vmovn_v),
  NEONMAP1(vmul_v, aarch64_neon_pmul, Add1ArgType),
  NEONMAP1(vmulq_v, aarch64_neon_pmul, Add1ArgType),
  NEONMAP1(vpadd_v, aarch64_neon_addp, Add1ArgType),
  NEONMAP2(vpaddl_v, aarch64_neon_uaddlp, aarch64_neon_saddlp, UnsignedAlts),
  NEONMAP2(vpaddlq_v, aarch64_neon_uaddlp, aarch64_neon_saddlp, UnsignedAlts),
  NEONMAP1(vpaddq_v, aarch64_neon_addp, Add1ArgType),
  NEONMAP1(vqabs_v, aarch64_neon_sqabs, Add1ArgType),
  NEONMAP1(vqabsq_v, aarch64_neon_sqabs, Add1ArgType),
  NEONMAP2(vqadd_v, aarch64_neon_uqadd, aarch64_neon_sqadd, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqaddq_v, aarch64_neon_uqadd, aarch64_neon_sqadd, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqdmlal_v, aarch64_neon_sqdmull, aarch64_neon_sqadd, 0),
  NEONMAP2(vqdmlsl_v, aarch64_neon_sqdmull, aarch64_neon_sqsub, 0),
  NEONMAP1(vqdmulh_lane_v, aarch64_neon_sqdmulh_lane, 0),
  NEONMAP1(vqdmulh_laneq_v, aarch64_neon_sqdmulh_laneq, 0),
  NEONMAP1(vqdmulh_v, aarch64_neon_sqdmulh, Add1ArgType),
  NEONMAP1(vqdmulhq_lane_v, aarch64_neon_sqdmulh_lane, 0),
  NEONMAP1(vqdmulhq_laneq_v, aarch64_neon_sqdmulh_laneq, 0),
  NEONMAP1(vqdmulhq_v, aarch64_neon_sqdmulh, Add1ArgType),
  NEONMAP1(vqdmull_v, aarch64_neon_sqdmull, Add1ArgType),
  NEONMAP2(vqmovn_v, aarch64_neon_uqxtn, aarch64_neon_sqxtn, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqmovun_v, aarch64_neon_sqxtun, Add1ArgType),
  NEONMAP1(vqneg_v, aarch64_neon_sqneg, Add1ArgType),
  NEONMAP1(vqnegq_v, aarch64_neon_sqneg, Add1ArgType),
  NEONMAP1(vqrdmlah_s16, aarch64_neon_sqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlah_s32, aarch64_neon_sqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlahq_s16, aarch64_neon_sqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlahq_s32, aarch64_neon_sqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlsh_s16, aarch64_neon_sqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlsh_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlshq_s16, aarch64_neon_sqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmlshq_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmulh_lane_v, aarch64_neon_sqrdmulh_lane, 0),
  NEONMAP1(vqrdmulh_laneq_v, aarch64_neon_sqrdmulh_laneq, 0),
  NEONMAP1(vqrdmulh_v, aarch64_neon_sqrdmulh, Add1ArgType),
  NEONMAP1(vqrdmulhq_lane_v, aarch64_neon_sqrdmulh_lane, 0),
  NEONMAP1(vqrdmulhq_laneq_v, aarch64_neon_sqrdmulh_laneq, 0),
  NEONMAP1(vqrdmulhq_v, aarch64_neon_sqrdmulh, Add1ArgType),
  NEONMAP2(vqrshl_v, aarch64_neon_uqrshl, aarch64_neon_sqrshl, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqrshlq_v, aarch64_neon_uqrshl, aarch64_neon_sqrshl, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshl_n_v, aarch64_neon_uqshl, aarch64_neon_sqshl, UnsignedAlts),
  NEONMAP2(vqshl_v, aarch64_neon_uqshl, aarch64_neon_sqshl, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshlq_n_v, aarch64_neon_uqshl, aarch64_neon_sqshl,UnsignedAlts),
  NEONMAP2(vqshlq_v, aarch64_neon_uqshl, aarch64_neon_sqshl, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqshlu_n_v, aarch64_neon_sqshlu, 0),
  NEONMAP1(vqshluq_n_v, aarch64_neon_sqshlu, 0),
  NEONMAP2(vqsub_v, aarch64_neon_uqsub, aarch64_neon_sqsub, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqsubq_v, aarch64_neon_uqsub, aarch64_neon_sqsub, Add1ArgType | UnsignedAlts),
  NEONMAP1(vraddhn_v, aarch64_neon_raddhn, Add1ArgType),
  NEONMAP1(vrax1q_u64, aarch64_crypto_rax1, 0),
  NEONMAP2(vrecpe_v, aarch64_neon_frecpe, aarch64_neon_urecpe, 0),
  NEONMAP2(vrecpeq_v, aarch64_neon_frecpe, aarch64_neon_urecpe, 0),
  NEONMAP1(vrecps_v, aarch64_neon_frecps, Add1ArgType),
  NEONMAP1(vrecpsq_v, aarch64_neon_frecps, Add1ArgType),
  NEONMAP2(vrhadd_v, aarch64_neon_urhadd, aarch64_neon_srhadd, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrhaddq_v, aarch64_neon_urhadd, aarch64_neon_srhadd, Add1ArgType | UnsignedAlts),
  NEONMAP1(vrnd32x_f32, aarch64_neon_frint32x, Add1ArgType),
  NEONMAP1(vrnd32x_f64, aarch64_neon_frint32x, Add1ArgType),
  NEONMAP1(vrnd32xq_f32, aarch64_neon_frint32x, Add1ArgType),
  NEONMAP1(vrnd32xq_f64, aarch64_neon_frint32x, Add1ArgType),
  NEONMAP1(vrnd32z_f32, aarch64_neon_frint32z, Add1ArgType),
  NEONMAP1(vrnd32z_f64, aarch64_neon_frint32z, Add1ArgType),
  NEONMAP1(vrnd32zq_f32, aarch64_neon_frint32z, Add1ArgType),
  NEONMAP1(vrnd32zq_f64, aarch64_neon_frint32z, Add1ArgType),
  NEONMAP1(vrnd64x_f32, aarch64_neon_frint64x, Add1ArgType),
  NEONMAP1(vrnd64x_f64, aarch64_neon_frint64x, Add1ArgType),
  NEONMAP1(vrnd64xq_f32, aarch64_neon_frint64x, Add1ArgType),
  NEONMAP1(vrnd64xq_f64, aarch64_neon_frint64x, Add1ArgType),
  NEONMAP1(vrnd64z_f32, aarch64_neon_frint64z, Add1ArgType),
  NEONMAP1(vrnd64z_f64, aarch64_neon_frint64z, Add1ArgType),
  NEONMAP1(vrnd64zq_f32, aarch64_neon_frint64z, Add1ArgType),
  NEONMAP1(vrnd64zq_f64, aarch64_neon_frint64z, Add1ArgType),
  NEONMAP0(vrndi_v),
  NEONMAP0(vrndiq_v),
  NEONMAP2(vrshl_v, aarch64_neon_urshl, aarch64_neon_srshl, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshlq_v, aarch64_neon_urshl, aarch64_neon_srshl, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshr_n_v, aarch64_neon_urshl, aarch64_neon_srshl, UnsignedAlts),
  NEONMAP2(vrshrq_n_v, aarch64_neon_urshl, aarch64_neon_srshl, UnsignedAlts),
  NEONMAP2(vrsqrte_v, aarch64_neon_frsqrte, aarch64_neon_ursqrte, 0),
  NEONMAP2(vrsqrteq_v, aarch64_neon_frsqrte, aarch64_neon_ursqrte, 0),
  NEONMAP1(vrsqrts_v, aarch64_neon_frsqrts, Add1ArgType),
  NEONMAP1(vrsqrtsq_v, aarch64_neon_frsqrts, Add1ArgType),
  NEONMAP1(vrsubhn_v, aarch64_neon_rsubhn, Add1ArgType),
  NEONMAP1(vsha1su0q_u32, aarch64_crypto_sha1su0, 0),
  NEONMAP1(vsha1su1q_u32, aarch64_crypto_sha1su1, 0),
  NEONMAP1(vsha256h2q_u32, aarch64_crypto_sha256h2, 0),
  NEONMAP1(vsha256hq_u32, aarch64_crypto_sha256h, 0),
  NEONMAP1(vsha256su0q_u32, aarch64_crypto_sha256su0, 0),
  NEONMAP1(vsha256su1q_u32, aarch64_crypto_sha256su1, 0),
  NEONMAP1(vsha512h2q_u64, aarch64_crypto_sha512h2, 0),
  NEONMAP1(vsha512hq_u64, aarch64_crypto_sha512h, 0),
  NEONMAP1(vsha512su0q_u64, aarch64_crypto_sha512su0, 0),
  NEONMAP1(vsha512su1q_u64, aarch64_crypto_sha512su1, 0),
  NEONMAP0(vshl_n_v),
  NEONMAP2(vshl_v, aarch64_neon_ushl, aarch64_neon_sshl, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshll_n_v),
  NEONMAP0(vshlq_n_v),
  NEONMAP2(vshlq_v, aarch64_neon_ushl, aarch64_neon_sshl, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshr_n_v),
  NEONMAP0(vshrn_n_v),
  NEONMAP0(vshrq_n_v),
  NEONMAP1(vsm3partw1q_u32, aarch64_crypto_sm3partw1, 0),
  NEONMAP1(vsm3partw2q_u32, aarch64_crypto_sm3partw2, 0),
  NEONMAP1(vsm3ss1q_u32, aarch64_crypto_sm3ss1, 0),
  NEONMAP1(vsm3tt1aq_u32, aarch64_crypto_sm3tt1a, 0),
  NEONMAP1(vsm3tt1bq_u32, aarch64_crypto_sm3tt1b, 0),
  NEONMAP1(vsm3tt2aq_u32, aarch64_crypto_sm3tt2a, 0),
  NEONMAP1(vsm3tt2bq_u32, aarch64_crypto_sm3tt2b, 0),
  NEONMAP1(vsm4ekeyq_u32, aarch64_crypto_sm4ekey, 0),
  NEONMAP1(vsm4eq_u32, aarch64_crypto_sm4e, 0),
  NEONMAP1(vst1_x2_v, aarch64_neon_st1x2, 0),
  NEONMAP1(vst1_x3_v, aarch64_neon_st1x3, 0),
  NEONMAP1(vst1_x4_v, aarch64_neon_st1x4, 0),
  NEONMAP1(vst1q_x2_v, aarch64_neon_st1x2, 0),
  NEONMAP1(vst1q_x3_v, aarch64_neon_st1x3, 0),
  NEONMAP1(vst1q_x4_v, aarch64_neon_st1x4, 0),
  NEONMAP0(vsubhn_v),
  NEONMAP0(vtst_v),
  NEONMAP0(vtstq_v),
  NEONMAP1(vusdot_s32, aarch64_neon_usdot, 0),
  NEONMAP1(vusdotq_s32, aarch64_neon_usdot, 0),
  NEONMAP1(vusmmlaq_s32, aarch64_neon_usmmla, 0),
  NEONMAP1(vxarq_u64, aarch64_crypto_xar, 0),
};

static const ARMVectorIntrinsicInfo AArch64SISDIntrinsicMap[] = {
  NEONMAP1(vabdd_f64, aarch64_sisd_fabd, Add1ArgType),
  NEONMAP1(vabds_f32, aarch64_sisd_fabd, Add1ArgType),
  NEONMAP1(vabsd_s64, aarch64_neon_abs, Add1ArgType),
  NEONMAP1(vaddlv_s32, aarch64_neon_saddlv, AddRetType | Add1ArgType),
  NEONMAP1(vaddlv_u32, aarch64_neon_uaddlv, AddRetType | Add1ArgType),
  NEONMAP1(vaddlvq_s32, aarch64_neon_saddlv, AddRetType | Add1ArgType),
  NEONMAP1(vaddlvq_u32, aarch64_neon_uaddlv, AddRetType | Add1ArgType),
  NEONMAP1(vaddv_f32, aarch64_neon_faddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddv_s32, aarch64_neon_saddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddv_u32, aarch64_neon_uaddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_f32, aarch64_neon_faddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_f64, aarch64_neon_faddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_s32, aarch64_neon_saddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_s64, aarch64_neon_saddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_u32, aarch64_neon_uaddv, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_u64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
  NEONMAP1(vcaged_f64, aarch64_neon_facge, AddRetType | Add1ArgType),
  NEONMAP1(vcages_f32, aarch64_neon_facge, AddRetType | Add1ArgType),
  NEONMAP1(vcagtd_f64, aarch64_neon_facgt, AddRetType | Add1ArgType),
  NEONMAP1(vcagts_f32, aarch64_neon_facgt, AddRetType | Add1ArgType),
  NEONMAP1(vcaled_f64, aarch64_neon_facge, AddRetType | Add1ArgType),
  NEONMAP1(vcales_f32, aarch64_neon_facge, AddRetType | Add1ArgType),
  NEONMAP1(vcaltd_f64, aarch64_neon_facgt, AddRetType | Add1ArgType),
  NEONMAP1(vcalts_f32, aarch64_neon_facgt, AddRetType | Add1ArgType),
  NEONMAP1(vcvtad_s64_f64, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
  NEONMAP1(vcvtad_u64_f64, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
  NEONMAP1(vcvtas_s32_f32, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
  NEONMAP1(vcvtas_u32_f32, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_n_f64_s64, aarch64_neon_vcvtfxs2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_n_f64_u64, aarch64_neon_vcvtfxu2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_n_s64_f64, aarch64_neon_vcvtfp2fxs, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_n_u64_f64, aarch64_neon_vcvtfp2fxu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_s64_f64, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
  NEONMAP1(vcvtd_u64_f64, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
  NEONMAP0(vcvth_bf16_f32),
  NEONMAP1(vcvtmd_s64_f64, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
  NEONMAP1(vcvtmd_u64_f64, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtms_s32_f32, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
  NEONMAP1(vcvtms_u32_f32, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnd_s64_f64, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnd_u64_f64, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtns_s32_f32, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
  NEONMAP1(vcvtns_u32_f32, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtpd_s64_f64, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
  NEONMAP1(vcvtpd_u64_f64, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtps_s32_f32, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
  NEONMAP1(vcvtps_u32_f32, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_n_f32_s32, aarch64_neon_vcvtfxs2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_n_f32_u32, aarch64_neon_vcvtfxu2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_n_s32_f32, aarch64_neon_vcvtfp2fxs, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_n_u32_f32, aarch64_neon_vcvtfp2fxu, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_s32_f32, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
  NEONMAP1(vcvts_u32_f32, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtxd_f32_f64, aarch64_sisd_fcvtxn, 0),
  NEONMAP1(vmaxnmv_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxnmvq_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxnmvq_f64, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxv_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxv_s32, aarch64_neon_smaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxv_u32, aarch64_neon_umaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxvq_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxvq_f64, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxvq_s32, aarch64_neon_smaxv, AddRetType | Add1ArgType),
  NEONMAP1(vmaxvq_u32, aarch64_neon_umaxv, AddRetType | Add1ArgType),
  NEONMAP1(vminnmv_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
  NEONMAP1(vminnmvq_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
  NEONMAP1(vminnmvq_f64, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
  NEONMAP1(vminv_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
  NEONMAP1(vminv_s32, aarch64_neon_sminv, AddRetType | Add1ArgType),
  NEONMAP1(vminv_u32, aarch64_neon_uminv, AddRetType | Add1ArgType),
  NEONMAP1(vminvq_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
  NEONMAP1(vminvq_f64, aarch64_neon_fminv, AddRetType | Add1ArgType),
  NEONMAP1(vminvq_s32, aarch64_neon_sminv, AddRetType | Add1ArgType),
  NEONMAP1(vminvq_u32, aarch64_neon_uminv, AddRetType | Add1ArgType),
  NEONMAP1(vmull_p64, aarch64_neon_pmull64, 0),
  NEONMAP1(vmulxd_f64, aarch64_neon_fmulx, Add1ArgType),
  NEONMAP1(vmulxs_f32, aarch64_neon_fmulx, Add1ArgType),
  NEONMAP1(vpaddd_s64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
  NEONMAP1(vpaddd_u64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxnmqd_f64, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxnms_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxqd_f64, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxs_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
  NEONMAP1(vpminnmqd_f64, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
  NEONMAP1(vpminnms_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
  NEONMAP1(vpminqd_f64, aarch64_neon_fminv, AddRetType | Add1ArgType),
  NEONMAP1(vpmins_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
  NEONMAP1(vqabsb_s8, aarch64_neon_sqabs, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqabsd_s64, aarch64_neon_sqabs, Add1ArgType),
  NEONMAP1(vqabsh_s16, aarch64_neon_sqabs, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqabss_s32, aarch64_neon_sqabs, Add1ArgType),
  NEONMAP1(vqaddb_s8, aarch64_neon_sqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqaddb_u8, aarch64_neon_uqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqaddd_s64, aarch64_neon_sqadd, Add1ArgType),
  NEONMAP1(vqaddd_u64, aarch64_neon_uqadd, Add1ArgType),
  NEONMAP1(vqaddh_s16, aarch64_neon_sqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqaddh_u16, aarch64_neon_uqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqadds_s32, aarch64_neon_sqadd, Add1ArgType),
  NEONMAP1(vqadds_u32, aarch64_neon_uqadd, Add1ArgType),
  NEONMAP1(vqdmulhh_s16, aarch64_neon_sqdmulh, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqdmulhs_s32, aarch64_neon_sqdmulh, Add1ArgType),
  NEONMAP1(vqdmullh_s16, aarch64_neon_sqdmull, VectorRet | Use128BitVectors),
  NEONMAP1(vqdmulls_s32, aarch64_neon_sqdmulls_scalar, 0),
  NEONMAP1(vqmovnd_s64, aarch64_neon_scalar_sqxtn, AddRetType | Add1ArgType),
  NEONMAP1(vqmovnd_u64, aarch64_neon_scalar_uqxtn, AddRetType | Add1ArgType),
  NEONMAP1(vqmovnh_s16, aarch64_neon_sqxtn, VectorRet | Use64BitVectors),
  NEONMAP1(vqmovnh_u16, aarch64_neon_uqxtn, VectorRet | Use64BitVectors),
  NEONMAP1(vqmovns_s32, aarch64_neon_sqxtn, VectorRet | Use64BitVectors),
  NEONMAP1(vqmovns_u32, aarch64_neon_uqxtn, VectorRet | Use64BitVectors),
  NEONMAP1(vqmovund_s64, aarch64_neon_scalar_sqxtun, AddRetType | Add1ArgType),
  NEONMAP1(vqmovunh_s16, aarch64_neon_sqxtun, VectorRet | Use64BitVectors),
  NEONMAP1(vqmovuns_s32, aarch64_neon_sqxtun, VectorRet | Use64BitVectors),
  NEONMAP1(vqnegb_s8, aarch64_neon_sqneg, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqnegd_s64, aarch64_neon_sqneg, Add1ArgType),
  NEONMAP1(vqnegh_s16, aarch64_neon_sqneg, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqnegs_s32, aarch64_neon_sqneg, Add1ArgType),
  NEONMAP1(vqrdmlahh_s16, aarch64_neon_sqrdmlah, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrdmlahs_s32, aarch64_neon_sqrdmlah, Add1ArgType),
  NEONMAP1(vqrdmlshh_s16, aarch64_neon_sqrdmlsh, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrdmlshs_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
  NEONMAP1(vqrdmulhh_s16, aarch64_neon_sqrdmulh, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrdmulhs_s32, aarch64_neon_sqrdmulh, Add1ArgType),
  NEONMAP1(vqrshlb_s8, aarch64_neon_sqrshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrshlb_u8, aarch64_neon_uqrshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrshld_s64, aarch64_neon_sqrshl, Add1ArgType),
  NEONMAP1(vqrshld_u64, aarch64_neon_uqrshl, Add1ArgType),
  NEONMAP1(vqrshlh_s16, aarch64_neon_sqrshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrshlh_u16, aarch64_neon_uqrshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqrshls_s32, aarch64_neon_sqrshl, Add1ArgType),
  NEONMAP1(vqrshls_u32, aarch64_neon_uqrshl, Add1ArgType),
  NEONMAP1(vqrshrnd_n_s64, aarch64_neon_sqrshrn, AddRetType),
  NEONMAP1(vqrshrnd_n_u64, aarch64_neon_uqrshrn, AddRetType),
  NEONMAP1(vqrshrnh_n_s16, aarch64_neon_sqrshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqrshrnh_n_u16, aarch64_neon_uqrshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqrshrns_n_s32, aarch64_neon_sqrshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqrshrns_n_u32, aarch64_neon_uqrshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqrshrund_n_s64, aarch64_neon_sqrshrun, AddRetType),
  NEONMAP1(vqrshrunh_n_s16, aarch64_neon_sqrshrun, VectorRet | Use64BitVectors),
  NEONMAP1(vqrshruns_n_s32, aarch64_neon_sqrshrun, VectorRet | Use64BitVectors),
  NEONMAP1(vqshlb_n_s8, aarch64_neon_sqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlb_n_u8, aarch64_neon_uqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlb_s8, aarch64_neon_sqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlb_u8, aarch64_neon_uqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshld_s64, aarch64_neon_sqshl, Add1ArgType),
  NEONMAP1(vqshld_u64, aarch64_neon_uqshl, Add1ArgType),
  NEONMAP1(vqshlh_n_s16, aarch64_neon_sqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlh_n_u16, aarch64_neon_uqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlh_s16, aarch64_neon_sqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlh_u16, aarch64_neon_uqshl, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshls_n_s32, aarch64_neon_sqshl, Add1ArgType),
  NEONMAP1(vqshls_n_u32, aarch64_neon_uqshl, Add1ArgType),
  NEONMAP1(vqshls_s32, aarch64_neon_sqshl, Add1ArgType),
  NEONMAP1(vqshls_u32, aarch64_neon_uqshl, Add1ArgType),
  NEONMAP1(vqshlub_n_s8, aarch64_neon_sqshlu, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshluh_n_s16, aarch64_neon_sqshlu, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqshlus_n_s32, aarch64_neon_sqshlu, Add1ArgType),
  NEONMAP1(vqshrnd_n_s64, aarch64_neon_sqshrn, AddRetType),
  NEONMAP1(vqshrnd_n_u64, aarch64_neon_uqshrn, AddRetType),
  NEONMAP1(vqshrnh_n_s16, aarch64_neon_sqshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqshrnh_n_u16, aarch64_neon_uqshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqshrns_n_s32, aarch64_neon_sqshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqshrns_n_u32, aarch64_neon_uqshrn, VectorRet | Use64BitVectors),
  NEONMAP1(vqshrund_n_s64, aarch64_neon_sqshrun, AddRetType),
  NEONMAP1(vqshrunh_n_s16, aarch64_neon_sqshrun, VectorRet | Use64BitVectors),
  NEONMAP1(vqshruns_n_s32, aarch64_neon_sqshrun, VectorRet | Use64BitVectors),
  NEONMAP1(vqsubb_s8, aarch64_neon_sqsub, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqsubb_u8, aarch64_neon_uqsub, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqsubd_s64, aarch64_neon_sqsub, Add1ArgType),
  NEONMAP1(vqsubd_u64, aarch64_neon_uqsub, Add1ArgType),
  NEONMAP1(vqsubh_s16, aarch64_neon_sqsub, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqsubh_u16, aarch64_neon_uqsub, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vqsubs_s32, aarch64_neon_sqsub, Add1ArgType),
  NEONMAP1(vqsubs_u32, aarch64_neon_uqsub, Add1ArgType),
  NEONMAP1(vrecped_f64, aarch64_neon_frecpe, Add1ArgType),
  NEONMAP1(vrecpes_f32, aarch64_neon_frecpe, Add1ArgType),
  NEONMAP1(vrecpxd_f64, aarch64_neon_frecpx, Add1ArgType),
  NEONMAP1(vrecpxs_f32, aarch64_neon_frecpx, Add1ArgType),
  NEONMAP1(vrshld_s64, aarch64_neon_srshl, Add1ArgType),
  NEONMAP1(vrshld_u64, aarch64_neon_urshl, Add1ArgType),
  NEONMAP1(vrsqrted_f64, aarch64_neon_frsqrte, Add1ArgType),
  NEONMAP1(vrsqrtes_f32, aarch64_neon_frsqrte, Add1ArgType),
  NEONMAP1(vrsqrtsd_f64, aarch64_neon_frsqrts, Add1ArgType),
  NEONMAP1(vrsqrtss_f32, aarch64_neon_frsqrts, Add1ArgType),
  NEONMAP1(vsha1cq_u32, aarch64_crypto_sha1c, 0),
  NEONMAP1(vsha1h_u32, aarch64_crypto_sha1h, 0),
  NEONMAP1(vsha1mq_u32, aarch64_crypto_sha1m, 0),
  NEONMAP1(vsha1pq_u32, aarch64_crypto_sha1p, 0),
  NEONMAP1(vshld_s64, aarch64_neon_sshl, Add1ArgType),
  NEONMAP1(vshld_u64, aarch64_neon_ushl, Add1ArgType),
  NEONMAP1(vslid_n_s64, aarch64_neon_vsli, Vectorize1ArgType),
  NEONMAP1(vslid_n_u64, aarch64_neon_vsli, Vectorize1ArgType),
  NEONMAP1(vsqaddb_u8, aarch64_neon_usqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vsqaddd_u64, aarch64_neon_usqadd, Add1ArgType),
  NEONMAP1(vsqaddh_u16, aarch64_neon_usqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vsqadds_u32, aarch64_neon_usqadd, Add1ArgType),
  NEONMAP1(vsrid_n_s64, aarch64_neon_vsri, Vectorize1ArgType),
  NEONMAP1(vsrid_n_u64, aarch64_neon_vsri, Vectorize1ArgType),
  NEONMAP1(vuqaddb_s8, aarch64_neon_suqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vuqaddd_s64, aarch64_neon_suqadd, Add1ArgType),
  NEONMAP1(vuqaddh_s16, aarch64_neon_suqadd, Vectorize1ArgType | Use64BitVectors),
  NEONMAP1(vuqadds_s32, aarch64_neon_suqadd, Add1ArgType),
  // FP16 scalar intrinisics go here.
  NEONMAP1(vabdh_f16, aarch64_sisd_fabd, Add1ArgType),
  NEONMAP1(vcvtah_s32_f16, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
  NEONMAP1(vcvtah_s64_f16, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
  NEONMAP1(vcvtah_u32_f16, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
  NEONMAP1(vcvtah_u64_f16, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_f16_s32, aarch64_neon_vcvtfxs2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_f16_s64, aarch64_neon_vcvtfxs2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_f16_u32, aarch64_neon_vcvtfxu2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_f16_u64, aarch64_neon_vcvtfxu2fp, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_s32_f16, aarch64_neon_vcvtfp2fxs, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_s64_f16, aarch64_neon_vcvtfp2fxs, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_u32_f16, aarch64_neon_vcvtfp2fxu, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_n_u64_f16, aarch64_neon_vcvtfp2fxu, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_s32_f16, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_s64_f16, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_u32_f16, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
  NEONMAP1(vcvth_u64_f16, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtmh_s32_f16, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
  NEONMAP1(vcvtmh_s64_f16, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
  NEONMAP1(vcvtmh_u32_f16, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtmh_u64_f16, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnh_s32_f16, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnh_s64_f16, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnh_u32_f16, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtnh_u64_f16, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtph_s32_f16, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
  NEONMAP1(vcvtph_s64_f16, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
  NEONMAP1(vcvtph_u32_f16, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
  NEONMAP1(vcvtph_u64_f16, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
  NEONMAP1(vmulxh_f16, aarch64_neon_fmulx, Add1ArgType),
  NEONMAP1(vrecpeh_f16, aarch64_neon_frecpe, Add1ArgType),
  NEONMAP1(vrecpxh_f16, aarch64_neon_frecpx, Add1ArgType),
  NEONMAP1(vrsqrteh_f16, aarch64_neon_frsqrte, Add1ArgType),
  NEONMAP1(vrsqrtsh_f16, aarch64_neon_frsqrts, Add1ArgType),
};

// Some intrinsics are equivalent for codegen.
static const std::pair<unsigned, unsigned> NEONEquivalentIntrinsicMap[] = {
  { NEON::BI__builtin_neon_splat_lane_bf16, NEON::BI__builtin_neon_splat_lane_v, },
  { NEON::BI__builtin_neon_splat_laneq_bf16, NEON::BI__builtin_neon_splat_laneq_v, },
  { NEON::BI__builtin_neon_splatq_lane_bf16, NEON::BI__builtin_neon_splatq_lane_v, },
  { NEON::BI__builtin_neon_splatq_laneq_bf16, NEON::BI__builtin_neon_splatq_laneq_v, },
  { NEON::BI__builtin_neon_vabd_f16, NEON::BI__builtin_neon_vabd_v, },
  { NEON::BI__builtin_neon_vabdq_f16, NEON::BI__builtin_neon_vabdq_v, },
  { NEON::BI__builtin_neon_vabs_f16, NEON::BI__builtin_neon_vabs_v, },
  { NEON::BI__builtin_neon_vabsq_f16, NEON::BI__builtin_neon_vabsq_v, },
  { NEON::BI__builtin_neon_vcage_f16, NEON::BI__builtin_neon_vcage_v, },
  { NEON::BI__builtin_neon_vcageq_f16, NEON::BI__builtin_neon_vcageq_v, },
  { NEON::BI__builtin_neon_vcagt_f16, NEON::BI__builtin_neon_vcagt_v, },
  { NEON::BI__builtin_neon_vcagtq_f16, NEON::BI__builtin_neon_vcagtq_v, },
  { NEON::BI__builtin_neon_vcale_f16, NEON::BI__builtin_neon_vcale_v, },
  { NEON::BI__builtin_neon_vcaleq_f16, NEON::BI__builtin_neon_vcaleq_v, },
  { NEON::BI__builtin_neon_vcalt_f16, NEON::BI__builtin_neon_vcalt_v, },
  { NEON::BI__builtin_neon_vcaltq_f16, NEON::BI__builtin_neon_vcaltq_v, },
  { NEON::BI__builtin_neon_vceqz_f16, NEON::BI__builtin_neon_vceqz_v, },
  { NEON::BI__builtin_neon_vceqzq_f16, NEON::BI__builtin_neon_vceqzq_v, },
  { NEON::BI__builtin_neon_vcgez_f16, NEON::BI__builtin_neon_vcgez_v, },
  { NEON::BI__builtin_neon_vcgezq_f16, NEON::BI__builtin_neon_vcgezq_v, },
  { NEON::BI__builtin_neon_vcgtz_f16, NEON::BI__builtin_neon_vcgtz_v, },
  { NEON::BI__builtin_neon_vcgtzq_f16, NEON::BI__builtin_neon_vcgtzq_v, },
  { NEON::BI__builtin_neon_vclez_f16, NEON::BI__builtin_neon_vclez_v, },
  { NEON::BI__builtin_neon_vclezq_f16, NEON::BI__builtin_neon_vclezq_v, },
  { NEON::BI__builtin_neon_vcltz_f16, NEON::BI__builtin_neon_vcltz_v, },
  { NEON::BI__builtin_neon_vcltzq_f16, NEON::BI__builtin_neon_vcltzq_v, },
  { NEON::BI__builtin_neon_vfma_f16, NEON::BI__builtin_neon_vfma_v, },
  { NEON::BI__builtin_neon_vfma_lane_f16, NEON::BI__builtin_neon_vfma_lane_v, },
  { NEON::BI__builtin_neon_vfma_laneq_f16, NEON::BI__builtin_neon_vfma_laneq_v, },
  { NEON::BI__builtin_neon_vfmaq_f16, NEON::BI__builtin_neon_vfmaq_v, },
  { NEON::BI__builtin_neon_vfmaq_lane_f16, NEON::BI__builtin_neon_vfmaq_lane_v, },
  { NEON::BI__builtin_neon_vfmaq_laneq_f16, NEON::BI__builtin_neon_vfmaq_laneq_v, },
  { NEON::BI__builtin_neon_vld1_bf16_x2, NEON::BI__builtin_neon_vld1_x2_v },
  { NEON::BI__builtin_neon_vld1_bf16_x3, NEON::BI__builtin_neon_vld1_x3_v },
  { NEON::BI__builtin_neon_vld1_bf16_x4, NEON::BI__builtin_neon_vld1_x4_v },
  { NEON::BI__builtin_neon_vld1_bf16, NEON::BI__builtin_neon_vld1_v },
  { NEON::BI__builtin_neon_vld1_dup_bf16, NEON::BI__builtin_neon_vld1_dup_v },
  { NEON::BI__builtin_neon_vld1_lane_bf16, NEON::BI__builtin_neon_vld1_lane_v },
  { NEON::BI__builtin_neon_vld1q_bf16_x2, NEON::BI__builtin_neon_vld1q_x2_v },
  { NEON::BI__builtin_neon_vld1q_bf16_x3, NEON::BI__builtin_neon_vld1q_x3_v },
  { NEON::BI__builtin_neon_vld1q_bf16_x4, NEON::BI__builtin_neon_vld1q_x4_v },
  { NEON::BI__builtin_neon_vld1q_bf16, NEON::BI__builtin_neon_vld1q_v },
  { NEON::BI__builtin_neon_vld1q_dup_bf16, NEON::BI__builtin_neon_vld1q_dup_v },
  { NEON::BI__builtin_neon_vld1q_lane_bf16, NEON::BI__builtin_neon_vld1q_lane_v },
  { NEON::BI__builtin_neon_vld2_bf16, NEON::BI__builtin_neon_vld2_v },
  { NEON::BI__builtin_neon_vld2_dup_bf16, NEON::BI__builtin_neon_vld2_dup_v },
  { NEON::BI__builtin_neon_vld2_lane_bf16, NEON::BI__builtin_neon_vld2_lane_v },
  { NEON::BI__builtin_neon_vld2q_bf16, NEON::BI__builtin_neon_vld2q_v },
  { NEON::BI__builtin_neon_vld2q_dup_bf16, NEON::BI__builtin_neon_vld2q_dup_v },
  { NEON::BI__builtin_neon_vld2q_lane_bf16, NEON::BI__builtin_neon_vld2q_lane_v },
  { NEON::BI__builtin_neon_vld3_bf16, NEON::BI__builtin_neon_vld3_v },
  { NEON::BI__builtin_neon_vld3_dup_bf16, NEON::BI__builtin_neon_vld3_dup_v },
  { NEON::BI__builtin_neon_vld3_lane_bf16, NEON::BI__builtin_neon_vld3_lane_v },
  { NEON::BI__builtin_neon_vld3q_bf16, NEON::BI__builtin_neon_vld3q_v },
  { NEON::BI__builtin_neon_vld3q_dup_bf16, NEON::BI__builtin_neon_vld3q_dup_v },
  { NEON::BI__builtin_neon_vld3q_lane_bf16, NEON::BI__builtin_neon_vld3q_lane_v },
  { NEON::BI__builtin_neon_vld4_bf16, NEON::BI__builtin_neon_vld4_v },
  { NEON::BI__builtin_neon_vld4_dup_bf16, NEON::BI__builtin_neon_vld4_dup_v },
  { NEON::BI__builtin_neon_vld4_lane_bf16, NEON::BI__builtin_neon_vld4_lane_v },
  { NEON::BI__builtin_neon_vld4q_bf16, NEON::BI__builtin_neon_vld4q_v },
  { NEON::BI__builtin_neon_vld4q_dup_bf16, NEON::BI__builtin_neon_vld4q_dup_v },
  { NEON::BI__builtin_neon_vld4q_lane_bf16, NEON::BI__builtin_neon_vld4q_lane_v },
  { NEON::BI__builtin_neon_vmax_f16, NEON::BI__builtin_neon_vmax_v, },
  { NEON::BI__builtin_neon_vmaxnm_f16, NEON::BI__builtin_neon_vmaxnm_v, },
  { NEON::BI__builtin_neon_vmaxnmq_f16, NEON::BI__builtin_neon_vmaxnmq_v, },
  { NEON::BI__builtin_neon_vmaxq_f16, NEON::BI__builtin_neon_vmaxq_v, },
  { NEON::BI__builtin_neon_vmin_f16, NEON::BI__builtin_neon_vmin_v, },
  { NEON::BI__builtin_neon_vminnm_f16, NEON::BI__builtin_neon_vminnm_v, },
  { NEON::BI__builtin_neon_vminnmq_f16, NEON::BI__builtin_neon_vminnmq_v, },
  { NEON::BI__builtin_neon_vminq_f16, NEON::BI__builtin_neon_vminq_v, },
  { NEON::BI__builtin_neon_vmulx_f16, NEON::BI__builtin_neon_vmulx_v, },
  { NEON::BI__builtin_neon_vmulxq_f16, NEON::BI__builtin_neon_vmulxq_v, },
  { NEON::BI__builtin_neon_vpadd_f16, NEON::BI__builtin_neon_vpadd_v, },
  { NEON::BI__builtin_neon_vpaddq_f16, NEON::BI__builtin_neon_vpaddq_v, },
  { NEON::BI__builtin_neon_vpmax_f16, NEON::BI__builtin_neon_vpmax_v, },
  { NEON::BI__builtin_neon_vpmaxnm_f16, NEON::BI__builtin_neon_vpmaxnm_v, },
  { NEON::BI__builtin_neon_vpmaxnmq_f16, NEON::BI__builtin_neon_vpmaxnmq_v, },
  { NEON::BI__builtin_neon_vpmaxq_f16, NEON::BI__builtin_neon_vpmaxq_v, },
  { NEON::BI__builtin_neon_vpmin_f16, NEON::BI__builtin_neon_vpmin_v, },
  { NEON::BI__builtin_neon_vpminnm_f16, NEON::BI__builtin_neon_vpminnm_v, },
  { NEON::BI__builtin_neon_vpminnmq_f16, NEON::BI__builtin_neon_vpminnmq_v, },
  { NEON::BI__builtin_neon_vpminq_f16, NEON::BI__builtin_neon_vpminq_v, },
  { NEON::BI__builtin_neon_vrecpe_f16, NEON::BI__builtin_neon_vrecpe_v, },
  { NEON::BI__builtin_neon_vrecpeq_f16, NEON::BI__builtin_neon_vrecpeq_v, },
  { NEON::BI__builtin_neon_vrecps_f16, NEON::BI__builtin_neon_vrecps_v, },
  { NEON::BI__builtin_neon_vrecpsq_f16, NEON::BI__builtin_neon_vrecpsq_v, },
  { NEON::BI__builtin_neon_vrnd_f16, NEON::BI__builtin_neon_vrnd_v, },
  { NEON::BI__builtin_neon_vrnda_f16, NEON::BI__builtin_neon_vrnda_v, },
  { NEON::BI__builtin_neon_vrndaq_f16, NEON::BI__builtin_neon_vrndaq_v, },
  { NEON::BI__builtin_neon_vrndi_f16, NEON::BI__builtin_neon_vrndi_v, },
  { NEON::BI__builtin_neon_vrndiq_f16, NEON::BI__builtin_neon_vrndiq_v, },
  { NEON::BI__builtin_neon_vrndm_f16, NEON::BI__builtin_neon_vrndm_v, },
  { NEON::BI__builtin_neon_vrndmq_f16, NEON::BI__builtin_neon_vrndmq_v, },
  { NEON::BI__builtin_neon_vrndn_f16, NEON::BI__builtin_neon_vrndn_v, },
  { NEON::BI__builtin_neon_vrndnq_f16, NEON::BI__builtin_neon_vrndnq_v, },
  { NEON::BI__builtin_neon_vrndp_f16, NEON::BI__builtin_neon_vrndp_v, },
  { NEON::BI__builtin_neon_vrndpq_f16, NEON::BI__builtin_neon_vrndpq_v, },
  { NEON::BI__builtin_neon_vrndq_f16, NEON::BI__builtin_neon_vrndq_v, },
  { NEON::BI__builtin_neon_vrndx_f16, NEON::BI__builtin_neon_vrndx_v, },
  { NEON::BI__builtin_neon_vrndxq_f16, NEON::BI__builtin_neon_vrndxq_v, },
  { NEON::BI__builtin_neon_vrsqrte_f16, NEON::BI__builtin_neon_vrsqrte_v, },
  { NEON::BI__builtin_neon_vrsqrteq_f16, NEON::BI__builtin_neon_vrsqrteq_v, },
  { NEON::BI__builtin_neon_vrsqrts_f16, NEON::BI__builtin_neon_vrsqrts_v, },
  { NEON::BI__builtin_neon_vrsqrtsq_f16, NEON::BI__builtin_neon_vrsqrtsq_v, },
  { NEON::BI__builtin_neon_vsqrt_f16, NEON::BI__builtin_neon_vsqrt_v, },
  { NEON::BI__builtin_neon_vsqrtq_f16, NEON::BI__builtin_neon_vsqrtq_v, },
  { NEON::BI__builtin_neon_vst1_bf16_x2, NEON::BI__builtin_neon_vst1_x2_v },
  { NEON::BI__builtin_neon_vst1_bf16_x3, NEON::BI__builtin_neon_vst1_x3_v },
  { NEON::BI__builtin_neon_vst1_bf16_x4, NEON::BI__builtin_neon_vst1_x4_v },
  { NEON::BI__builtin_neon_vst1_bf16, NEON::BI__builtin_neon_vst1_v },
  { NEON::BI__builtin_neon_vst1_lane_bf16, NEON::BI__builtin_neon_vst1_lane_v },
  { NEON::BI__builtin_neon_vst1q_bf16_x2, NEON::BI__builtin_neon_vst1q_x2_v },
  { NEON::BI__builtin_neon_vst1q_bf16_x3, NEON::BI__builtin_neon_vst1q_x3_v },
  { NEON::BI__builtin_neon_vst1q_bf16_x4, NEON::BI__builtin_neon_vst1q_x4_v },
  { NEON::BI__builtin_neon_vst1q_bf16, NEON::BI__builtin_neon_vst1q_v },
  { NEON::BI__builtin_neon_vst1q_lane_bf16, NEON::BI__builtin_neon_vst1q_lane_v },
  { NEON::BI__builtin_neon_vst2_bf16, NEON::BI__builtin_neon_vst2_v },
  { NEON::BI__builtin_neon_vst2_lane_bf16, NEON::BI__builtin_neon_vst2_lane_v },
  { NEON::BI__builtin_neon_vst2q_bf16, NEON::BI__builtin_neon_vst2q_v },
  { NEON::BI__builtin_neon_vst2q_lane_bf16, NEON::BI__builtin_neon_vst2q_lane_v },
  { NEON::BI__builtin_neon_vst3_bf16, NEON::BI__builtin_neon_vst3_v },
  { NEON::BI__builtin_neon_vst3_lane_bf16, NEON::BI__builtin_neon_vst3_lane_v },
  { NEON::BI__builtin_neon_vst3q_bf16, NEON::BI__builtin_neon_vst3q_v },
  { NEON::BI__builtin_neon_vst3q_lane_bf16, NEON::BI__builtin_neon_vst3q_lane_v },
  { NEON::BI__builtin_neon_vst4_bf16, NEON::BI__builtin_neon_vst4_v },
  { NEON::BI__builtin_neon_vst4_lane_bf16, NEON::BI__builtin_neon_vst4_lane_v },
  { NEON::BI__builtin_neon_vst4q_bf16, NEON::BI__builtin_neon_vst4q_v },
  { NEON::BI__builtin_neon_vst4q_lane_bf16, NEON::BI__builtin_neon_vst4q_lane_v },
  // The mangling rules cause us to have one ID for each type for vldap1(q)_lane
  // and vstl1(q)_lane, but codegen is equivalent for all of them. Choose an
  // arbitrary one to be handled as tha canonical variation.
  { NEON::BI__builtin_neon_vldap1_lane_u64, NEON::BI__builtin_neon_vldap1_lane_s64 },
  { NEON::BI__builtin_neon_vldap1_lane_f64, NEON::BI__builtin_neon_vldap1_lane_s64 },
  { NEON::BI__builtin_neon_vldap1_lane_p64, NEON::BI__builtin_neon_vldap1_lane_s64 },
  { NEON::BI__builtin_neon_vldap1q_lane_u64, NEON::BI__builtin_neon_vldap1q_lane_s64 },
  { NEON::BI__builtin_neon_vldap1q_lane_f64, NEON::BI__builtin_neon_vldap1q_lane_s64 },
  { NEON::BI__builtin_neon_vldap1q_lane_p64, NEON::BI__builtin_neon_vldap1q_lane_s64 },
  { NEON::BI__builtin_neon_vstl1_lane_u64, NEON::BI__builtin_neon_vstl1_lane_s64 },
  { NEON::BI__builtin_neon_vstl1_lane_f64, NEON::BI__builtin_neon_vstl1_lane_s64 },
  { NEON::BI__builtin_neon_vstl1_lane_p64, NEON::BI__builtin_neon_vstl1_lane_s64 },
  { NEON::BI__builtin_neon_vstl1q_lane_u64, NEON::BI__builtin_neon_vstl1q_lane_s64 },
  { NEON::BI__builtin_neon_vstl1q_lane_f64, NEON::BI__builtin_neon_vstl1q_lane_s64 },
  { NEON::BI__builtin_neon_vstl1q_lane_p64, NEON::BI__builtin_neon_vstl1q_lane_s64 },
};

#undef NEONMAP0
#undef NEONMAP1
#undef NEONMAP2

#define SVEMAP1(NameBase, LLVMIntrinsic, TypeModifier)                         \
  {                                                                            \
    #NameBase, SVE::BI__builtin_sve_##NameBase, Intrinsic::LLVMIntrinsic, 0,   \
        TypeModifier                                                           \
  }

#define SVEMAP2(NameBase, TypeModifier)                                        \
  { #NameBase, SVE::BI__builtin_sve_##NameBase, 0, 0, TypeModifier }
static const ARMVectorIntrinsicInfo AArch64SVEIntrinsicMap[] = {
#define GET_SVE_LLVM_INTRINSIC_MAP
#include "clang/Basic/arm_sve_builtin_cg.inc"
#include "clang/Basic/BuiltinsAArch64NeonSVEBridge_cg.def"
#undef GET_SVE_LLVM_INTRINSIC_MAP
};

#undef SVEMAP1
#undef SVEMAP2

#define SMEMAP1(NameBase, LLVMIntrinsic, TypeModifier)                         \
  {                                                                            \
    #NameBase, SME::BI__builtin_sme_##NameBase, Intrinsic::LLVMIntrinsic, 0,   \
        TypeModifier                                                           \
  }

#define SMEMAP2(NameBase, TypeModifier)                                        \
  { #NameBase, SME::BI__builtin_sme_##NameBase, 0, 0, TypeModifier }
static const ARMVectorIntrinsicInfo AArch64SMEIntrinsicMap[] = {
#define GET_SME_LLVM_INTRINSIC_MAP
#include "clang/Basic/arm_sme_builtin_cg.inc"
#undef GET_SME_LLVM_INTRINSIC_MAP
};

#undef SMEMAP1
#undef SMEMAP2

static bool NEONSIMDIntrinsicsProvenSorted = false;

static bool AArch64SIMDIntrinsicsProvenSorted = false;
static bool AArch64SISDIntrinsicsProvenSorted = false;
static bool AArch64SVEIntrinsicsProvenSorted = false;
static bool AArch64SMEIntrinsicsProvenSorted = false;

static const ARMVectorIntrinsicInfo *
findARMVectorIntrinsicInMap(ArrayRef<ARMVectorIntrinsicInfo> IntrinsicMap,
                            unsigned BuiltinID, bool &MapProvenSorted) {

#ifndef NDEBUG
  if (!MapProvenSorted) {
    assert(llvm::is_sorted(IntrinsicMap));
    MapProvenSorted = true;
  }
#endif

  const ARMVectorIntrinsicInfo *Builtin =
      llvm::lower_bound(IntrinsicMap, BuiltinID);

  if (Builtin != IntrinsicMap.end() && Builtin->BuiltinID == BuiltinID)
    return Builtin;

  return nullptr;
}

Function *CodeGenFunction::LookupNeonLLVMIntrinsic(unsigned IntrinsicID,
                                                   unsigned Modifier,
                                                   llvm::Type *ArgType,
                                                   const CallExpr *E) {
  int VectorSize = 0;
  if (Modifier & Use64BitVectors)
    VectorSize = 64;
  else if (Modifier & Use128BitVectors)
    VectorSize = 128;

  // Return type.
  SmallVector<llvm::Type *, 3> Tys;
  if (Modifier & AddRetType) {
    llvm::Type *Ty = ConvertType(E->getCallReturnType(getContext()));
    if (Modifier & VectorizeRetType)
      Ty = llvm::FixedVectorType::get(
          Ty, VectorSize ? VectorSize / Ty->getPrimitiveSizeInBits() : 1);

    Tys.push_back(Ty);
  }

  // Arguments.
  if (Modifier & VectorizeArgTypes) {
    int Elts = VectorSize ? VectorSize / ArgType->getPrimitiveSizeInBits() : 1;
    ArgType = llvm::FixedVectorType::get(ArgType, Elts);
  }

  if (Modifier & (Add1ArgType | Add2ArgTypes))
    Tys.push_back(ArgType);

  if (Modifier & Add2ArgTypes)
    Tys.push_back(ArgType);

  if (Modifier & InventFloatType)
    Tys.push_back(FloatTy);

  return CGM.getIntrinsic(IntrinsicID, Tys);
}

static Value *EmitCommonNeonSISDBuiltinExpr(
    CodeGenFunction &CGF, const ARMVectorIntrinsicInfo &SISDInfo,
    SmallVectorImpl<Value *> &Ops, const CallExpr *E) {
  unsigned BuiltinID = SISDInfo.BuiltinID;
  unsigned int Int = SISDInfo.LLVMIntrinsic;
  unsigned Modifier = SISDInfo.TypeModifier;
  const char *s = SISDInfo.NameHint;

  switch (BuiltinID) {
  case NEON::BI__builtin_neon_vcled_s64:
  case NEON::BI__builtin_neon_vcled_u64:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_s64:
  case NEON::BI__builtin_neon_vcltd_u64:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcales_f32:
  case NEON::BI__builtin_neon_vcaled_f64:
  case NEON::BI__builtin_neon_vcalts_f32:
  case NEON::BI__builtin_neon_vcaltd_f64:
    // Only one direction of comparisons actually exist, cmle is actually a cmge
    // with swapped operands. The table gives us the right intrinsic but we
    // still need to do the swap.
    std::swap(Ops[0], Ops[1]);
    break;
  }

  assert(Int && "Generic code assumes a valid intrinsic");

  // Determine the type(s) of this overloaded AArch64 intrinsic.
  const Expr *Arg = E->getArg(0);
  llvm::Type *ArgTy = CGF.ConvertType(Arg->getType());
  Function *F = CGF.LookupNeonLLVMIntrinsic(Int, Modifier, ArgTy, E);

  int j = 0;
  ConstantInt *C0 = ConstantInt::get(CGF.SizeTy, 0);
  for (Function::const_arg_iterator ai = F->arg_begin(), ae = F->arg_end();
       ai != ae; ++ai, ++j) {
    llvm::Type *ArgTy = ai->getType();
    if (Ops[j]->getType()->getPrimitiveSizeInBits() ==
             ArgTy->getPrimitiveSizeInBits())
      continue;

    assert(ArgTy->isVectorTy() && !Ops[j]->getType()->isVectorTy());
    // The constant argument to an _n_ intrinsic always has Int32Ty, so truncate
    // it before inserting.
    Ops[j] = CGF.Builder.CreateTruncOrBitCast(
        Ops[j], cast<llvm::VectorType>(ArgTy)->getElementType());
    Ops[j] =
        CGF.Builder.CreateInsertElement(PoisonValue::get(ArgTy), Ops[j], C0);
  }

  Value *Result = CGF.EmitNeonCall(F, Ops, s);
  llvm::Type *ResultType = CGF.ConvertType(E->getType());
  if (ResultType->getPrimitiveSizeInBits().getFixedValue() <
      Result->getType()->getPrimitiveSizeInBits().getFixedValue())
    return CGF.Builder.CreateExtractElement(Result, C0);

  return CGF.Builder.CreateBitCast(Result, ResultType, s);
}

Value *CodeGenFunction::EmitCommonNeonBuiltinExpr(
    unsigned BuiltinID, unsigned LLVMIntrinsic, unsigned AltLLVMIntrinsic,
    const char *NameHint, unsigned Modifier, const CallExpr *E,
    SmallVectorImpl<llvm::Value *> &Ops, Address PtrOp0, Address PtrOp1,
    llvm::Triple::ArchType Arch) {
  // Get the last argument, which specifies the vector type.
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  std::optional<llvm::APSInt> NeonTypeConst =
      Arg->getIntegerConstantExpr(getContext());
  if (!NeonTypeConst)
    return nullptr;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(NeonTypeConst->getZExtValue());
  const bool Usgn = Type.isUnsigned();
  const bool Quad = Type.isQuad();
  const bool Floating = Type.isFloatingPoint();
  const bool HasFastHalfType = getTarget().hasFastHalfType();
  const bool AllowBFloatArgsAndRet =
      getTargetHooks().getABIInfo().allowBFloatArgsAndRet();

  llvm::FixedVectorType *VTy =
      GetNeonType(this, Type, HasFastHalfType, false, AllowBFloatArgsAndRet);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return nullptr;

  auto getAlignmentValue32 = [&](Address addr) -> Value* {
    return Builder.getInt32(addr.getAlignment().getQuantity());
  };

  unsigned Int = LLVMIntrinsic;
  if ((Modifier & UnsignedAlts) && !Usgn)
    Int = AltLLVMIntrinsic;

  switch (BuiltinID) {
  default: break;
  case NEON::BI__builtin_neon_splat_lane_v:
  case NEON::BI__builtin_neon_splat_laneq_v:
  case NEON::BI__builtin_neon_splatq_lane_v:
  case NEON::BI__builtin_neon_splatq_laneq_v: {
    auto NumElements = VTy->getElementCount();
    if (BuiltinID == NEON::BI__builtin_neon_splatq_lane_v)
      NumElements = NumElements * 2;
    if (BuiltinID == NEON::BI__builtin_neon_splat_laneq_v)
      NumElements = NumElements.divideCoefficientBy(2);

    Ops[0] = Builder.CreateBitCast(Ops[0], VTy);
    return EmitNeonSplat(Ops[0], cast<ConstantInt>(Ops[1]), NumElements);
  }
  case NEON::BI__builtin_neon_vpadd_v:
  case NEON::BI__builtin_neon_vpaddq_v:
    // We don't allow fp/int overloading of intrinsics.
    if (VTy->getElementType()->isFloatingPointTy() &&
        Int == Intrinsic::aarch64_neon_addp)
      Int = Intrinsic::aarch64_neon_faddp;
    break;
  case NEON::BI__builtin_neon_vabs_v:
  case NEON::BI__builtin_neon_vabsq_v:
    if (VTy->getElementType()->isFloatingPointTy())
      return EmitNeonCall(CGM.getIntrinsic(Intrinsic::fabs, Ty), Ops, "vabs");
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Ty), Ops, "vabs");
  case NEON::BI__builtin_neon_vadd_v:
  case NEON::BI__builtin_neon_vaddq_v: {
    llvm::Type *VTy = llvm::FixedVectorType::get(Int8Ty, Quad ? 16 : 8);
    Ops[0] = Builder.CreateBitCast(Ops[0], VTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], VTy);
    Ops[0] =  Builder.CreateXor(Ops[0], Ops[1]);
    return Builder.CreateBitCast(Ops[0], Ty);
  }
  case NEON::BI__builtin_neon_vaddhn_v: {
    llvm::FixedVectorType *SrcTy =
        llvm::FixedVectorType::getExtendedElementVectorType(VTy);

    // %sum = add <4 x i32> %lhs, %rhs
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], SrcTy);
    Ops[0] = Builder.CreateAdd(Ops[0], Ops[1], "vaddhn");

    // %high = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
    Constant *ShiftAmt =
        ConstantInt::get(SrcTy, SrcTy->getScalarSizeInBits() / 2);
    Ops[0] = Builder.CreateLShr(Ops[0], ShiftAmt, "vaddhn");

    // %res = trunc <4 x i32> %high to <4 x i16>
    return Builder.CreateTrunc(Ops[0], VTy, "vaddhn");
  }
  case NEON::BI__builtin_neon_vcale_v:
  case NEON::BI__builtin_neon_vcaleq_v:
  case NEON::BI__builtin_neon_vcalt_v:
  case NEON::BI__builtin_neon_vcaltq_v:
    std::swap(Ops[0], Ops[1]);
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcage_v:
  case NEON::BI__builtin_neon_vcageq_v:
  case NEON::BI__builtin_neon_vcagt_v:
  case NEON::BI__builtin_neon_vcagtq_v: {
    llvm::Type *Ty;
    switch (VTy->getScalarSizeInBits()) {
    default: llvm_unreachable("unexpected type");
    case 32:
      Ty = FloatTy;
      break;
    case 64:
      Ty = DoubleTy;
      break;
    case 16:
      Ty = HalfTy;
      break;
    }
    auto *VecFlt = llvm::FixedVectorType::get(Ty, VTy->getNumElements());
    llvm::Type *Tys[] = { VTy, VecFlt };
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    return EmitNeonCall(F, Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vceqz_v:
  case NEON::BI__builtin_neon_vceqzq_v:
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], Ty, Floating ? ICmpInst::FCMP_OEQ : ICmpInst::ICMP_EQ, "vceqz");
  case NEON::BI__builtin_neon_vcgez_v:
  case NEON::BI__builtin_neon_vcgezq_v:
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], Ty, Floating ? ICmpInst::FCMP_OGE : ICmpInst::ICMP_SGE,
        "vcgez");
  case NEON::BI__builtin_neon_vclez_v:
  case NEON::BI__builtin_neon_vclezq_v:
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], Ty, Floating ? ICmpInst::FCMP_OLE : ICmpInst::ICMP_SLE,
        "vclez");
  case NEON::BI__builtin_neon_vcgtz_v:
  case NEON::BI__builtin_neon_vcgtzq_v:
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], Ty, Floating ? ICmpInst::FCMP_OGT : ICmpInst::ICMP_SGT,
        "vcgtz");
  case NEON::BI__builtin_neon_vcltz_v:
  case NEON::BI__builtin_neon_vcltzq_v:
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], Ty, Floating ? ICmpInst::FCMP_OLT : ICmpInst::ICMP_SLT,
        "vcltz");
  case NEON::BI__builtin_neon_vclz_v:
  case NEON::BI__builtin_neon_vclzq_v:
    // We generate target-independent intrinsic, which needs a second argument
    // for whether or not clz of zero is undefined; on ARM it isn't.
    Ops.push_back(Builder.getInt1(getTarget().isCLZForZeroUndef()));
    break;
  case NEON::BI__builtin_neon_vcvt_f32_v:
  case NEON::BI__builtin_neon_vcvtq_f32_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, Quad),
                     HasFastHalfType);
    return Usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  case NEON::BI__builtin_neon_vcvt_f16_s16:
  case NEON::BI__builtin_neon_vcvt_f16_u16:
  case NEON::BI__builtin_neon_vcvtq_f16_s16:
  case NEON::BI__builtin_neon_vcvtq_f16_u16:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float16, false, Quad),
                     HasFastHalfType);
    return Usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  case NEON::BI__builtin_neon_vcvt_n_f16_s16:
  case NEON::BI__builtin_neon_vcvt_n_f16_u16:
  case NEON::BI__builtin_neon_vcvtq_n_f16_s16:
  case NEON::BI__builtin_neon_vcvtq_n_f16_u16: {
    llvm::Type *Tys[2] = { GetFloatNeonType(this, Type), Ty };
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case NEON::BI__builtin_neon_vcvt_n_f32_v:
  case NEON::BI__builtin_neon_vcvt_n_f64_v:
  case NEON::BI__builtin_neon_vcvtq_n_f32_v:
  case NEON::BI__builtin_neon_vcvtq_n_f64_v: {
    llvm::Type *Tys[2] = { GetFloatNeonType(this, Type), Ty };
    Int = Usgn ? LLVMIntrinsic : AltLLVMIntrinsic;
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case NEON::BI__builtin_neon_vcvt_n_s16_f16:
  case NEON::BI__builtin_neon_vcvt_n_s32_v:
  case NEON::BI__builtin_neon_vcvt_n_u16_f16:
  case NEON::BI__builtin_neon_vcvt_n_u32_v:
  case NEON::BI__builtin_neon_vcvt_n_s64_v:
  case NEON::BI__builtin_neon_vcvt_n_u64_v:
  case NEON::BI__builtin_neon_vcvtq_n_s16_f16:
  case NEON::BI__builtin_neon_vcvtq_n_s32_v:
  case NEON::BI__builtin_neon_vcvtq_n_u16_f16:
  case NEON::BI__builtin_neon_vcvtq_n_u32_v:
  case NEON::BI__builtin_neon_vcvtq_n_s64_v:
  case NEON::BI__builtin_neon_vcvtq_n_u64_v: {
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case NEON::BI__builtin_neon_vcvt_s32_v:
  case NEON::BI__builtin_neon_vcvt_u32_v:
  case NEON::BI__builtin_neon_vcvt_s64_v:
  case NEON::BI__builtin_neon_vcvt_u64_v:
  case NEON::BI__builtin_neon_vcvt_s16_f16:
  case NEON::BI__builtin_neon_vcvt_u16_f16:
  case NEON::BI__builtin_neon_vcvtq_s32_v:
  case NEON::BI__builtin_neon_vcvtq_u32_v:
  case NEON::BI__builtin_neon_vcvtq_s64_v:
  case NEON::BI__builtin_neon_vcvtq_u64_v:
  case NEON::BI__builtin_neon_vcvtq_s16_f16:
  case NEON::BI__builtin_neon_vcvtq_u16_f16: {
    Ops[0] = Builder.CreateBitCast(Ops[0], GetFloatNeonType(this, Type));
    return Usgn ? Builder.CreateFPToUI(Ops[0], Ty, "vcvt")
                : Builder.CreateFPToSI(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvta_s16_f16:
  case NEON::BI__builtin_neon_vcvta_s32_v:
  case NEON::BI__builtin_neon_vcvta_s64_v:
  case NEON::BI__builtin_neon_vcvta_u16_f16:
  case NEON::BI__builtin_neon_vcvta_u32_v:
  case NEON::BI__builtin_neon_vcvta_u64_v:
  case NEON::BI__builtin_neon_vcvtaq_s16_f16:
  case NEON::BI__builtin_neon_vcvtaq_s32_v:
  case NEON::BI__builtin_neon_vcvtaq_s64_v:
  case NEON::BI__builtin_neon_vcvtaq_u16_f16:
  case NEON::BI__builtin_neon_vcvtaq_u32_v:
  case NEON::BI__builtin_neon_vcvtaq_u64_v:
  case NEON::BI__builtin_neon_vcvtn_s16_f16:
  case NEON::BI__builtin_neon_vcvtn_s32_v:
  case NEON::BI__builtin_neon_vcvtn_s64_v:
  case NEON::BI__builtin_neon_vcvtn_u16_f16:
  case NEON::BI__builtin_neon_vcvtn_u32_v:
  case NEON::BI__builtin_neon_vcvtn_u64_v:
  case NEON::BI__builtin_neon_vcvtnq_s16_f16:
  case NEON::BI__builtin_neon_vcvtnq_s32_v:
  case NEON::BI__builtin_neon_vcvtnq_s64_v:
  case NEON::BI__builtin_neon_vcvtnq_u16_f16:
  case NEON::BI__builtin_neon_vcvtnq_u32_v:
  case NEON::BI__builtin_neon_vcvtnq_u64_v:
  case NEON::BI__builtin_neon_vcvtp_s16_f16:
  case NEON::BI__builtin_neon_vcvtp_s32_v:
  case NEON::BI__builtin_neon_vcvtp_s64_v:
  case NEON::BI__builtin_neon_vcvtp_u16_f16:
  case NEON::BI__builtin_neon_vcvtp_u32_v:
  case NEON::BI__builtin_neon_vcvtp_u64_v:
  case NEON::BI__builtin_neon_vcvtpq_s16_f16:
  case NEON::BI__builtin_neon_vcvtpq_s32_v:
  case NEON::BI__builtin_neon_vcvtpq_s64_v:
  case NEON::BI__builtin_neon_vcvtpq_u16_f16:
  case NEON::BI__builtin_neon_vcvtpq_u32_v:
  case NEON::BI__builtin_neon_vcvtpq_u64_v:
  case NEON::BI__builtin_neon_vcvtm_s16_f16:
  case NEON::BI__builtin_neon_vcvtm_s32_v:
  case NEON::BI__builtin_neon_vcvtm_s64_v:
  case NEON::BI__builtin_neon_vcvtm_u16_f16:
  case NEON::BI__builtin_neon_vcvtm_u32_v:
  case NEON::BI__builtin_neon_vcvtm_u64_v:
  case NEON::BI__builtin_neon_vcvtmq_s16_f16:
  case NEON::BI__builtin_neon_vcvtmq_s32_v:
  case NEON::BI__builtin_neon_vcvtmq_s64_v:
  case NEON::BI__builtin_neon_vcvtmq_u16_f16:
  case NEON::BI__builtin_neon_vcvtmq_u32_v:
  case NEON::BI__builtin_neon_vcvtmq_u64_v: {
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vcvtx_f32_v: {
    llvm::Type *Tys[2] = { VTy->getTruncatedElementVectorType(VTy), Ty};
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, NameHint);

  }
  case NEON::BI__builtin_neon_vext_v:
  case NEON::BI__builtin_neon_vextq_v: {
    int CV = cast<ConstantInt>(Ops[2])->getSExtValue();
    SmallVector<int, 16> Indices;
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
      Indices.push_back(i+CV);

    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    return Builder.CreateShuffleVector(Ops[0], Ops[1], Indices, "vext");
  }
  case NEON::BI__builtin_neon_vfma_v:
  case NEON::BI__builtin_neon_vfmaq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);

    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, Ty,
        {Ops[1], Ops[2], Ops[0]});
  }
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v: {
    llvm::Type *Tys[] = {Ty, Int8PtrTy};
    Ops.push_back(getAlignmentValue32(PtrOp0));
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, "vld1");
  }
  case NEON::BI__builtin_neon_vld1_x2_v:
  case NEON::BI__builtin_neon_vld1q_x2_v:
  case NEON::BI__builtin_neon_vld1_x3_v:
  case NEON::BI__builtin_neon_vld1q_x3_v:
  case NEON::BI__builtin_neon_vld1_x4_v:
  case NEON::BI__builtin_neon_vld1q_x4_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld1xN");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v:
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v:
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v:
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v:
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v:
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v: {
    llvm::Type *Tys[] = {Ty, Int8PtrTy};
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    Value *Align = getAlignmentValue32(PtrOp1);
    Ops[1] = Builder.CreateCall(F, {Ops[1], Align}, NameHint);
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v: {
    Value *V = PoisonValue::get(Ty);
    PtrOp0 = PtrOp0.withElementType(VTy->getElementType());
    LoadInst *Ld = Builder.CreateLoad(PtrOp0);
    llvm::Constant *CI = ConstantInt::get(SizeTy, 0);
    Ops[0] = Builder.CreateInsertElement(V, Ld, CI);
    return EmitNeonSplat(Ops[0], CI);
  }
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v:
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v:
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v: {
    llvm::Type *Tys[] = {Ty, Int8PtrTy};
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    for (unsigned I = 2; I < Ops.size() - 1; ++I)
      Ops[I] = Builder.CreateBitCast(Ops[I], Ty);
    Ops.push_back(getAlignmentValue32(PtrOp1));
    Ops[1] = Builder.CreateCall(F, ArrayRef(Ops).slice(1), NameHint);
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vmovl_v: {
    llvm::FixedVectorType *DTy =
        llvm::FixedVectorType::getTruncatedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], DTy);
    if (Usgn)
      return Builder.CreateZExt(Ops[0], Ty, "vmovl");
    return Builder.CreateSExt(Ops[0], Ty, "vmovl");
  }
  case NEON::BI__builtin_neon_vmovn_v: {
    llvm::FixedVectorType *QTy =
        llvm::FixedVectorType::getExtendedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], QTy);
    return Builder.CreateTrunc(Ops[0], Ty, "vmovn");
  }
  case NEON::BI__builtin_neon_vmull_v:
    // FIXME: the integer vmull operations could be emitted in terms of pure
    // LLVM IR (2 exts followed by a mul). Unfortunately LLVM has a habit of
    // hoisting the exts outside loops. Until global ISel comes along that can
    // see through such movement this leads to bad CodeGen. So we need an
    // intrinsic for now.
    Int = Usgn ? Intrinsic::arm_neon_vmullu : Intrinsic::arm_neon_vmulls;
    Int = Type.isPoly() ? (unsigned)Intrinsic::arm_neon_vmullp : Int;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmull");
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v: {
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy =
      llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    auto *NarrowTy =
        llvm::FixedVectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vpaddl_v:
  case NEON::BI__builtin_neon_vpaddlq_v: {
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy = llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    auto *NarrowTy =
        llvm::FixedVectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vpaddl");
  }
  case NEON::BI__builtin_neon_vqdmlal_v:
  case NEON::BI__builtin_neon_vqdmlsl_v: {
    SmallVector<Value *, 2> MulOps(Ops.begin() + 1, Ops.end());
    Ops[1] =
        EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Ty), MulOps, "vqdmlal");
    Ops.resize(2);
    return EmitNeonCall(CGM.getIntrinsic(AltLLVMIntrinsic, Ty), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vqdmulhq_lane_v:
  case NEON::BI__builtin_neon_vqdmulh_lane_v:
  case NEON::BI__builtin_neon_vqrdmulhq_lane_v:
  case NEON::BI__builtin_neon_vqrdmulh_lane_v: {
    auto *RTy = cast<llvm::FixedVectorType>(Ty);
    if (BuiltinID == NEON::BI__builtin_neon_vqdmulhq_lane_v ||
        BuiltinID == NEON::BI__builtin_neon_vqrdmulhq_lane_v)
      RTy = llvm::FixedVectorType::get(RTy->getElementType(),
                                       RTy->getNumElements() * 2);
    llvm::Type *Tys[2] = {
        RTy, GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                             /*isQuad*/ false))};
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vqdmulhq_laneq_v:
  case NEON::BI__builtin_neon_vqdmulh_laneq_v:
  case NEON::BI__builtin_neon_vqrdmulhq_laneq_v:
  case NEON::BI__builtin_neon_vqrdmulh_laneq_v: {
    llvm::Type *Tys[2] = {
        Ty, GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                            /*isQuad*/ true))};
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vqshl_n_v:
  case NEON::BI__builtin_neon_vqshlq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshl_n",
                        1, false);
  case NEON::BI__builtin_neon_vqshlu_n_v:
  case NEON::BI__builtin_neon_vqshluq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshlu_n",
                        1, false);
  case NEON::BI__builtin_neon_vrecpe_v:
  case NEON::BI__builtin_neon_vrecpeq_v:
  case NEON::BI__builtin_neon_vrsqrte_v:
  case NEON::BI__builtin_neon_vrsqrteq_v:
    Int = Ty->isFPOrFPVectorTy() ? LLVMIntrinsic : AltLLVMIntrinsic;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, NameHint);
  case NEON::BI__builtin_neon_vrndi_v:
  case NEON::BI__builtin_neon_vrndiq_v:
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_nearbyint
              : Intrinsic::nearbyint;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, NameHint);
  case NEON::BI__builtin_neon_vrshr_n_v:
  case NEON::BI__builtin_neon_vrshrq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshr_n",
                        1, true);
  case NEON::BI__builtin_neon_vsha512hq_u64:
  case NEON::BI__builtin_neon_vsha512h2q_u64:
  case NEON::BI__builtin_neon_vsha512su0q_u64:
  case NEON::BI__builtin_neon_vsha512su1q_u64: {
    Function *F = CGM.getIntrinsic(Int);
    return EmitNeonCall(F, Ops, "");
  }
  case NEON::BI__builtin_neon_vshl_n_v:
  case NEON::BI__builtin_neon_vshlq_n_v:
    Ops[1] = EmitNeonShiftVector(Ops[1], Ty, false);
    return Builder.CreateShl(Builder.CreateBitCast(Ops[0],Ty), Ops[1],
                             "vshl_n");
  case NEON::BI__builtin_neon_vshll_n_v: {
    llvm::FixedVectorType *SrcTy =
        llvm::FixedVectorType::getTruncatedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    if (Usgn)
      Ops[0] = Builder.CreateZExt(Ops[0], VTy);
    else
      Ops[0] = Builder.CreateSExt(Ops[0], VTy);
    Ops[1] = EmitNeonShiftVector(Ops[1], VTy, false);
    return Builder.CreateShl(Ops[0], Ops[1], "vshll_n");
  }
  case NEON::BI__builtin_neon_vshrn_n_v: {
    llvm::FixedVectorType *SrcTy =
        llvm::FixedVectorType::getExtendedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = EmitNeonShiftVector(Ops[1], SrcTy, false);
    if (Usgn)
      Ops[0] = Builder.CreateLShr(Ops[0], Ops[1]);
    else
      Ops[0] = Builder.CreateAShr(Ops[0], Ops[1]);
    return Builder.CreateTrunc(Ops[0], Ty, "vshrn_n");
  }
  case NEON::BI__builtin_neon_vshr_n_v:
  case NEON::BI__builtin_neon_vshrq_n_v:
    return EmitNeonRShiftImm(Ops[0], Ops[1], Ty, Usgn, "vshr_n");
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v:
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v:
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v:
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v:
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v:
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v:
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v: {
    llvm::Type *Tys[] = {Int8PtrTy, Ty};
    Ops.push_back(getAlignmentValue32(PtrOp0));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "");
  }
  case NEON::BI__builtin_neon_vsm3partw1q_u32:
  case NEON::BI__builtin_neon_vsm3partw2q_u32:
  case NEON::BI__builtin_neon_vsm3ss1q_u32:
  case NEON::BI__builtin_neon_vsm4ekeyq_u32:
  case NEON::BI__builtin_neon_vsm4eq_u32: {
    Function *F = CGM.getIntrinsic(Int);
    return EmitNeonCall(F, Ops, "");
  }
  case NEON::BI__builtin_neon_vsm3tt1aq_u32:
  case NEON::BI__builtin_neon_vsm3tt1bq_u32:
  case NEON::BI__builtin_neon_vsm3tt2aq_u32:
  case NEON::BI__builtin_neon_vsm3tt2bq_u32: {
    Function *F = CGM.getIntrinsic(Int);
    Ops[3] = Builder.CreateZExt(Ops[3], Int64Ty);
    return EmitNeonCall(F, Ops, "");
  }
  case NEON::BI__builtin_neon_vst1_x2_v:
  case NEON::BI__builtin_neon_vst1q_x2_v:
  case NEON::BI__builtin_neon_vst1_x3_v:
  case NEON::BI__builtin_neon_vst1q_x3_v:
  case NEON::BI__builtin_neon_vst1_x4_v:
  case NEON::BI__builtin_neon_vst1q_x4_v: {
    // TODO: Currently in AArch32 mode the pointer operand comes first, whereas
    // in AArch64 it comes last. We may want to stick to one or another.
    if (Arch == llvm::Triple::aarch64 || Arch == llvm::Triple::aarch64_be ||
        Arch == llvm::Triple::aarch64_32) {
      llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
      std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
      return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, "");
    }
    llvm::Type *Tys[2] = {UnqualPtrTy, VTy};
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, "");
  }
  case NEON::BI__builtin_neon_vsubhn_v: {
    llvm::FixedVectorType *SrcTy =
        llvm::FixedVectorType::getExtendedElementVectorType(VTy);

    // %sum = add <4 x i32> %lhs, %rhs
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], SrcTy);
    Ops[0] = Builder.CreateSub(Ops[0], Ops[1], "vsubhn");

    // %high = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
    Constant *ShiftAmt =
        ConstantInt::get(SrcTy, SrcTy->getScalarSizeInBits() / 2);
    Ops[0] = Builder.CreateLShr(Ops[0], ShiftAmt, "vsubhn");

    // %res = trunc <4 x i32> %high to <4 x i16>
    return Builder.CreateTrunc(Ops[0], VTy, "vsubhn");
  }
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(i+vi);
        Indices.push_back(i+e+vi);
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vtrn");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vtst_v:
  case NEON::BI__builtin_neon_vtstq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[0] = Builder.CreateAnd(Ops[0], Ops[1]);
    Ops[0] = Builder.CreateICmp(ICmpInst::ICMP_NE, Ops[0],
                                ConstantAggregateZero::get(Ty));
    return Builder.CreateSExt(Ops[0], Ty, "vtst");
  }
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
        Indices.push_back(2*i+vi);

      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vuzp");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vxarq_u64: {
    Function *F = CGM.getIntrinsic(Int);
    Ops[2] = Builder.CreateZExt(Ops[2], Int64Ty);
    return EmitNeonCall(F, Ops, "");
  }
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back((i + vi*e) >> 1);
        Indices.push_back(((i + vi*e) >> 1)+e);
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vzip");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vdot_s32:
  case NEON::BI__builtin_neon_vdot_u32:
  case NEON::BI__builtin_neon_vdotq_s32:
  case NEON::BI__builtin_neon_vdotq_u32: {
    auto *InputTy =
        llvm::FixedVectorType::get(Int8Ty, Ty->getPrimitiveSizeInBits() / 8);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vdot");
  }
  case NEON::BI__builtin_neon_vfmlal_low_f16:
  case NEON::BI__builtin_neon_vfmlalq_low_f16: {
    auto *InputTy =
        llvm::FixedVectorType::get(HalfTy, Ty->getPrimitiveSizeInBits() / 16);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vfmlal_low");
  }
  case NEON::BI__builtin_neon_vfmlsl_low_f16:
  case NEON::BI__builtin_neon_vfmlslq_low_f16: {
    auto *InputTy =
        llvm::FixedVectorType::get(HalfTy, Ty->getPrimitiveSizeInBits() / 16);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vfmlsl_low");
  }
  case NEON::BI__builtin_neon_vfmlal_high_f16:
  case NEON::BI__builtin_neon_vfmlalq_high_f16: {
    auto *InputTy =
        llvm::FixedVectorType::get(HalfTy, Ty->getPrimitiveSizeInBits() / 16);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vfmlal_high");
  }
  case NEON::BI__builtin_neon_vfmlsl_high_f16:
  case NEON::BI__builtin_neon_vfmlslq_high_f16: {
    auto *InputTy =
        llvm::FixedVectorType::get(HalfTy, Ty->getPrimitiveSizeInBits() / 16);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vfmlsl_high");
  }
  case NEON::BI__builtin_neon_vmmlaq_s32:
  case NEON::BI__builtin_neon_vmmlaq_u32: {
    auto *InputTy =
        llvm::FixedVectorType::get(Int8Ty, Ty->getPrimitiveSizeInBits() / 8);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, "vmmla");
  }
  case NEON::BI__builtin_neon_vusmmlaq_s32: {
    auto *InputTy =
        llvm::FixedVectorType::get(Int8Ty, Ty->getPrimitiveSizeInBits() / 8);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vusmmla");
  }
  case NEON::BI__builtin_neon_vusdot_s32:
  case NEON::BI__builtin_neon_vusdotq_s32: {
    auto *InputTy =
        llvm::FixedVectorType::get(Int8Ty, Ty->getPrimitiveSizeInBits() / 8);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vusdot");
  }
  case NEON::BI__builtin_neon_vbfdot_f32:
  case NEON::BI__builtin_neon_vbfdotq_f32: {
    llvm::Type *InputTy =
        llvm::FixedVectorType::get(BFloatTy, Ty->getPrimitiveSizeInBits() / 16);
    llvm::Type *Tys[2] = { Ty, InputTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vbfdot");
  }
  case NEON::BI__builtin_neon___a32_vcvt_bf16_f32: {
    llvm::Type *Tys[1] = { Ty };
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvtfp2bf");
  }

  }

  assert(Int && "Expected valid intrinsic number");

  // Determine the type(s) of this overloaded AArch64 intrinsic.
  Function *F = LookupNeonLLVMIntrinsic(Int, Modifier, Ty, E);

  Value *Result = EmitNeonCall(F, Ops, NameHint);
  llvm::Type *ResultType = ConvertType(E->getType());
  // AArch64 intrinsic one-element vector type cast to
  // scalar type expected by the builtin
  return Builder.CreateBitCast(Result, ResultType, NameHint);
}

Value *
CodeGenFunction::EmitAArch64CompareBuiltinExpr(Value *Op, llvm::Type *Ty,
                                               const CmpInst::Predicate Pred,
                                               const Twine &Name) {

  if (isa<FixedVectorType>(Ty)) {
    // Vector types are cast to i8 vectors. Recover original type.
    Op = Builder.CreateBitCast(Op, Ty);
  }

  if (CmpInst::isFPPredicate(Pred)) {
    if (Pred == CmpInst::FCMP_OEQ)
      Op = Builder.CreateFCmp(Pred, Op, Constant::getNullValue(Op->getType()));
    else
      Op = Builder.CreateFCmpS(Pred, Op, Constant::getNullValue(Op->getType()));
  } else {
    Op = Builder.CreateICmp(Pred, Op, Constant::getNullValue(Op->getType()));
  }

  llvm::Type *ResTy = Ty;
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty))
    ResTy = FixedVectorType::get(
        IntegerType::get(getLLVMContext(), VTy->getScalarSizeInBits()),
        VTy->getNumElements());

  return Builder.CreateSExt(Op, ResTy, Name);
}

static Value *packTBLDVectorList(CodeGenFunction &CGF, ArrayRef<Value *> Ops,
                                 Value *ExtOp, Value *IndexOp,
                                 llvm::Type *ResTy, unsigned IntID,
                                 const char *Name) {
  SmallVector<Value *, 2> TblOps;
  if (ExtOp)
    TblOps.push_back(ExtOp);

  // Build a vector containing sequential number like (0, 1, 2, ..., 15)
  SmallVector<int, 16> Indices;
  auto *TblTy = cast<llvm::FixedVectorType>(Ops[0]->getType());
  for (unsigned i = 0, e = TblTy->getNumElements(); i != e; ++i) {
    Indices.push_back(2*i);
    Indices.push_back(2*i+1);
  }

  int PairPos = 0, End = Ops.size() - 1;
  while (PairPos < End) {
    TblOps.push_back(CGF.Builder.CreateShuffleVector(Ops[PairPos],
                                                     Ops[PairPos+1], Indices,
                                                     Name));
    PairPos += 2;
  }

  // If there's an odd number of 64-bit lookup table, fill the high 64-bit
  // of the 128-bit lookup table with zero.
  if (PairPos == End) {
    Value *ZeroTbl = ConstantAggregateZero::get(TblTy);
    TblOps.push_back(CGF.Builder.CreateShuffleVector(Ops[PairPos],
                                                     ZeroTbl, Indices, Name));
  }

  Function *TblF;
  TblOps.push_back(IndexOp);
  TblF = CGF.CGM.getIntrinsic(IntID, ResTy);

  return CGF.EmitNeonCall(TblF, TblOps, Name);
}

Value *CodeGenFunction::GetValueForARMHint(unsigned BuiltinID) {
  unsigned Value;
  switch (BuiltinID) {
  default:
    return nullptr;
  case clang::ARM::BI__builtin_arm_nop:
    Value = 0;
    break;
  case clang::ARM::BI__builtin_arm_yield:
  case clang::ARM::BI__yield:
    Value = 1;
    break;
  case clang::ARM::BI__builtin_arm_wfe:
  case clang::ARM::BI__wfe:
    Value = 2;
    break;
  case clang::ARM::BI__builtin_arm_wfi:
  case clang::ARM::BI__wfi:
    Value = 3;
    break;
  case clang::ARM::BI__builtin_arm_sev:
  case clang::ARM::BI__sev:
    Value = 4;
    break;
  case clang::ARM::BI__builtin_arm_sevl:
  case clang::ARM::BI__sevl:
    Value = 5;
    break;
  }

  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_hint),
                            llvm::ConstantInt::get(Int32Ty, Value));
}

enum SpecialRegisterAccessKind {
  NormalRead,
  VolatileRead,
  Write,
};

// Generates the IR for the read/write special register builtin,
// ValueType is the type of the value that is to be written or read,
// RegisterType is the type of the register being written to or read from.
static Value *EmitSpecialRegisterBuiltin(CodeGenFunction &CGF,
                                         const CallExpr *E,
                                         llvm::Type *RegisterType,
                                         llvm::Type *ValueType,
                                         SpecialRegisterAccessKind AccessKind,
                                         StringRef SysReg = "") {
  // write and register intrinsics only support 32, 64 and 128 bit operations.
  assert((RegisterType->isIntegerTy(32) || RegisterType->isIntegerTy(64) ||
          RegisterType->isIntegerTy(128)) &&
         "Unsupported size for register.");

  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  CodeGen::CodeGenModule &CGM = CGF.CGM;
  LLVMContext &Context = CGM.getLLVMContext();

  if (SysReg.empty()) {
    const Expr *SysRegStrExpr = E->getArg(0)->IgnoreParenCasts();
    SysReg = cast<clang::StringLiteral>(SysRegStrExpr)->getString();
  }

  llvm::Metadata *Ops[] = { llvm::MDString::get(Context, SysReg) };
  llvm::MDNode *RegName = llvm::MDNode::get(Context, Ops);
  llvm::Value *Metadata = llvm::MetadataAsValue::get(Context, RegName);

  llvm::Type *Types[] = { RegisterType };

  bool MixedTypes = RegisterType->isIntegerTy(64) && ValueType->isIntegerTy(32);
  assert(!(RegisterType->isIntegerTy(32) && ValueType->isIntegerTy(64))
            && "Can't fit 64-bit value in 32-bit register");

  if (AccessKind != Write) {
    assert(AccessKind == NormalRead || AccessKind == VolatileRead);
    llvm::Function *F = CGM.getIntrinsic(
        AccessKind == VolatileRead ? Intrinsic::read_volatile_register
                                   : Intrinsic::read_register,
        Types);
    llvm::Value *Call = Builder.CreateCall(F, Metadata);

    if (MixedTypes)
      // Read into 64 bit register and then truncate result to 32 bit.
      return Builder.CreateTrunc(Call, ValueType);

    if (ValueType->isPointerTy())
      // Have i32/i64 result (Call) but want to return a VoidPtrTy (i8*).
      return Builder.CreateIntToPtr(Call, ValueType);

    return Call;
  }

  llvm::Function *F = CGM.getIntrinsic(Intrinsic::write_register, Types);
  llvm::Value *ArgValue = CGF.EmitScalarExpr(E->getArg(1));
  if (MixedTypes) {
    // Extend 32 bit write value to 64 bit to pass to write.
    ArgValue = Builder.CreateZExt(ArgValue, RegisterType);
    return Builder.CreateCall(F, { Metadata, ArgValue });
  }

  if (ValueType->isPointerTy()) {
    // Have VoidPtrTy ArgValue but want to return an i32/i64.
    ArgValue = Builder.CreatePtrToInt(ArgValue, RegisterType);
    return Builder.CreateCall(F, { Metadata, ArgValue });
  }

  return Builder.CreateCall(F, { Metadata, ArgValue });
}

/// Return true if BuiltinID is an overloaded Neon intrinsic with an extra
/// argument that specifies the vector type.
static bool HasExtraNeonArgument(unsigned BuiltinID) {
  switch (BuiltinID) {
  default: break;
  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vget_lane_mf8:
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vgetq_lane_mf8:
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_mf8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_mf8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
  case NEON::BI__builtin_neon_vsha1h_u32:
  case NEON::BI__builtin_neon_vsha1cq_u32:
  case NEON::BI__builtin_neon_vsha1pq_u32:
  case NEON::BI__builtin_neon_vsha1mq_u32:
  case NEON::BI__builtin_neon_vcvth_bf16_f32:
  case clang::ARM::BI_MoveToCoprocessor:
  case clang::ARM::BI_MoveToCoprocessor2:
    return false;
  }
  return true;
}

Value *CodeGenFunction::EmitARMBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E,
                                           ReturnValueSlot ReturnValue,
                                           llvm::Triple::ArchType Arch) {
  if (auto Hint = GetValueForARMHint(BuiltinID))
    return Hint;

  if (BuiltinID == clang::ARM::BI__emit) {
    bool IsThumb = getTarget().getTriple().getArch() == llvm::Triple::thumb;
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(VoidTy, /*Variadic=*/false);

    Expr::EvalResult Result;
    if (!E->getArg(0)->EvaluateAsInt(Result, CGM.getContext()))
      llvm_unreachable("Sema will ensure that the parameter is constant");

    llvm::APSInt Value = Result.Val.getInt();
    uint64_t ZExtValue = Value.zextOrTrunc(IsThumb ? 16 : 32).getZExtValue();

    llvm::InlineAsm *Emit =
        IsThumb ? InlineAsm::get(FTy, ".inst.n 0x" + utohexstr(ZExtValue), "",
                                 /*hasSideEffects=*/true)
                : InlineAsm::get(FTy, ".inst 0x" + utohexstr(ZExtValue), "",
                                 /*hasSideEffects=*/true);

    return Builder.CreateCall(Emit);
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_dbg) {
    Value *Option = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_dbg), Option);
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_prefetch) {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *RW      = EmitScalarExpr(E->getArg(1));
    Value *IsData  = EmitScalarExpr(E->getArg(2));

    // Locality is not supported on ARM target
    Value *Locality = llvm::ConstantInt::get(Int32Ty, 3);

    Function *F = CGM.getIntrinsic(Intrinsic::prefetch, Address->getType());
    return Builder.CreateCall(F, {Address, RW, Locality, IsData});
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_rbit) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::bitreverse, Arg->getType()), Arg, "rbit");
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_clz ||
      BuiltinID == clang::ARM::BI__builtin_arm_clz64) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Arg->getType());
    Value *Res = Builder.CreateCall(F, {Arg, Builder.getInt1(false)});
    if (BuiltinID == clang::ARM::BI__builtin_arm_clz64)
      Res = Builder.CreateTrunc(Res, Builder.getInt32Ty());
    return Res;
  }


  if (BuiltinID == clang::ARM::BI__builtin_arm_cls) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cls), Arg, "cls");
  }
  if (BuiltinID == clang::ARM::BI__builtin_arm_cls64) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cls64), Arg,
                              "cls");
  }

  if (BuiltinID == clang::ARM::BI__clear_cache) {
    assert(E->getNumArgs() == 2 && "__clear_cache takes 2 arguments");
    const FunctionDecl *FD = E->getDirectCallee();
    Value *Ops[2];
    for (unsigned i = 0; i < 2; i++)
      Ops[i] = EmitScalarExpr(E->getArg(i));
    llvm::Type *Ty = CGM.getTypes().ConvertType(FD->getType());
    llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    StringRef Name = FD->getName();
    return EmitNounwindRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Ops);
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_mcrr ||
      BuiltinID == clang::ARM::BI__builtin_arm_mcrr2) {
    Function *F;

    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin");
    case clang::ARM::BI__builtin_arm_mcrr:
      F = CGM.getIntrinsic(Intrinsic::arm_mcrr);
      break;
    case clang::ARM::BI__builtin_arm_mcrr2:
      F = CGM.getIntrinsic(Intrinsic::arm_mcrr2);
      break;
    }

    // MCRR{2} instruction has 5 operands but
    // the intrinsic has 4 because Rt and Rt2
    // are represented as a single unsigned 64
    // bit integer in the intrinsic definition
    // but internally it's represented as 2 32
    // bit integers.

    Value *Coproc = EmitScalarExpr(E->getArg(0));
    Value *Opc1 = EmitScalarExpr(E->getArg(1));
    Value *RtAndRt2 = EmitScalarExpr(E->getArg(2));
    Value *CRm = EmitScalarExpr(E->getArg(3));

    Value *C1 = llvm::ConstantInt::get(Int64Ty, 32);
    Value *Rt = Builder.CreateTruncOrBitCast(RtAndRt2, Int32Ty);
    Value *Rt2 = Builder.CreateLShr(RtAndRt2, C1);
    Rt2 = Builder.CreateTruncOrBitCast(Rt2, Int32Ty);

    return Builder.CreateCall(F, {Coproc, Opc1, Rt, Rt2, CRm});
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_mrrc ||
      BuiltinID == clang::ARM::BI__builtin_arm_mrrc2) {
    Function *F;

    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin");
    case clang::ARM::BI__builtin_arm_mrrc:
      F = CGM.getIntrinsic(Intrinsic::arm_mrrc);
      break;
    case clang::ARM::BI__builtin_arm_mrrc2:
      F = CGM.getIntrinsic(Intrinsic::arm_mrrc2);
      break;
    }

    Value *Coproc = EmitScalarExpr(E->getArg(0));
    Value *Opc1 = EmitScalarExpr(E->getArg(1));
    Value *CRm  = EmitScalarExpr(E->getArg(2));
    Value *RtAndRt2 = Builder.CreateCall(F, {Coproc, Opc1, CRm});

    // Returns an unsigned 64 bit integer, represented
    // as two 32 bit integers.

    Value *Rt = Builder.CreateExtractValue(RtAndRt2, 1);
    Value *Rt1 = Builder.CreateExtractValue(RtAndRt2, 0);
    Rt = Builder.CreateZExt(Rt, Int64Ty);
    Rt1 = Builder.CreateZExt(Rt1, Int64Ty);

    Value *ShiftCast = llvm::ConstantInt::get(Int64Ty, 32);
    RtAndRt2 = Builder.CreateShl(Rt, ShiftCast, "shl", true);
    RtAndRt2 = Builder.CreateOr(RtAndRt2, Rt1);

    return Builder.CreateBitCast(RtAndRt2, ConvertType(E->getType()));
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_ldrexd ||
      ((BuiltinID == clang::ARM::BI__builtin_arm_ldrex ||
        BuiltinID == clang::ARM::BI__builtin_arm_ldaex) &&
       getContext().getTypeSize(E->getType()) == 64) ||
      BuiltinID == clang::ARM::BI__ldrexd) {
    Function *F;

    switch (BuiltinID) {
    default: llvm_unreachable("unexpected builtin");
    case clang::ARM::BI__builtin_arm_ldaex:
      F = CGM.getIntrinsic(Intrinsic::arm_ldaexd);
      break;
    case clang::ARM::BI__builtin_arm_ldrexd:
    case clang::ARM::BI__builtin_arm_ldrex:
    case clang::ARM::BI__ldrexd:
      F = CGM.getIntrinsic(Intrinsic::arm_ldrexd);
      break;
    }

    Value *LdPtr = EmitScalarExpr(E->getArg(0));
    Value *Val = Builder.CreateCall(F, LdPtr, "ldrexd");

    Value *Val0 = Builder.CreateExtractValue(Val, 1);
    Value *Val1 = Builder.CreateExtractValue(Val, 0);
    Val0 = Builder.CreateZExt(Val0, Int64Ty);
    Val1 = Builder.CreateZExt(Val1, Int64Ty);

    Value *ShiftCst = llvm::ConstantInt::get(Int64Ty, 32);
    Val = Builder.CreateShl(Val0, ShiftCst, "shl", true /* nuw */);
    Val = Builder.CreateOr(Val, Val1);
    return Builder.CreateBitCast(Val, ConvertType(E->getType()));
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_ldrex ||
      BuiltinID == clang::ARM::BI__builtin_arm_ldaex) {
    Value *LoadAddr = EmitScalarExpr(E->getArg(0));

    QualType Ty = E->getType();
    llvm::Type *RealResTy = ConvertType(Ty);
    llvm::Type *IntTy =
        llvm::IntegerType::get(getLLVMContext(), getContext().getTypeSize(Ty));

    Function *F = CGM.getIntrinsic(
        BuiltinID == clang::ARM::BI__builtin_arm_ldaex ? Intrinsic::arm_ldaex
                                                       : Intrinsic::arm_ldrex,
        UnqualPtrTy);
    CallInst *Val = Builder.CreateCall(F, LoadAddr, "ldrex");
    Val->addParamAttr(
        0, Attribute::get(getLLVMContext(), Attribute::ElementType, IntTy));

    if (RealResTy->isPointerTy())
      return Builder.CreateIntToPtr(Val, RealResTy);
    else {
      llvm::Type *IntResTy = llvm::IntegerType::get(
          getLLVMContext(), CGM.getDataLayout().getTypeSizeInBits(RealResTy));
      return Builder.CreateBitCast(Builder.CreateTruncOrBitCast(Val, IntResTy),
                                   RealResTy);
    }
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_strexd ||
      ((BuiltinID == clang::ARM::BI__builtin_arm_stlex ||
        BuiltinID == clang::ARM::BI__builtin_arm_strex) &&
       getContext().getTypeSize(E->getArg(0)->getType()) == 64)) {
    Function *F = CGM.getIntrinsic(
        BuiltinID == clang::ARM::BI__builtin_arm_stlex ? Intrinsic::arm_stlexd
                                                       : Intrinsic::arm_strexd);
    llvm::Type *STy = llvm::StructType::get(Int32Ty, Int32Ty);

    Address Tmp = CreateMemTemp(E->getArg(0)->getType());
    Value *Val = EmitScalarExpr(E->getArg(0));
    Builder.CreateStore(Val, Tmp);

    Address LdPtr = Tmp.withElementType(STy);
    Val = Builder.CreateLoad(LdPtr);

    Value *Arg0 = Builder.CreateExtractValue(Val, 0);
    Value *Arg1 = Builder.CreateExtractValue(Val, 1);
    Value *StPtr = EmitScalarExpr(E->getArg(1));
    return Builder.CreateCall(F, {Arg0, Arg1, StPtr}, "strexd");
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_strex ||
      BuiltinID == clang::ARM::BI__builtin_arm_stlex) {
    Value *StoreVal = EmitScalarExpr(E->getArg(0));
    Value *StoreAddr = EmitScalarExpr(E->getArg(1));

    QualType Ty = E->getArg(0)->getType();
    llvm::Type *StoreTy =
        llvm::IntegerType::get(getLLVMContext(), getContext().getTypeSize(Ty));

    if (StoreVal->getType()->isPointerTy())
      StoreVal = Builder.CreatePtrToInt(StoreVal, Int32Ty);
    else {
      llvm::Type *IntTy = llvm::IntegerType::get(
          getLLVMContext(),
          CGM.getDataLayout().getTypeSizeInBits(StoreVal->getType()));
      StoreVal = Builder.CreateBitCast(StoreVal, IntTy);
      StoreVal = Builder.CreateZExtOrBitCast(StoreVal, Int32Ty);
    }

    Function *F = CGM.getIntrinsic(
        BuiltinID == clang::ARM::BI__builtin_arm_stlex ? Intrinsic::arm_stlex
                                                       : Intrinsic::arm_strex,
        StoreAddr->getType());

    CallInst *CI = Builder.CreateCall(F, {StoreVal, StoreAddr}, "strex");
    CI->addParamAttr(
        1, Attribute::get(getLLVMContext(), Attribute::ElementType, StoreTy));
    return CI;
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_clrex) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_clrex);
    return Builder.CreateCall(F);
  }

  // CRC32
  Intrinsic::ID CRCIntrinsicID = Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case clang::ARM::BI__builtin_arm_crc32b:
    CRCIntrinsicID = Intrinsic::arm_crc32b; break;
  case clang::ARM::BI__builtin_arm_crc32cb:
    CRCIntrinsicID = Intrinsic::arm_crc32cb; break;
  case clang::ARM::BI__builtin_arm_crc32h:
    CRCIntrinsicID = Intrinsic::arm_crc32h; break;
  case clang::ARM::BI__builtin_arm_crc32ch:
    CRCIntrinsicID = Intrinsic::arm_crc32ch; break;
  case clang::ARM::BI__builtin_arm_crc32w:
  case clang::ARM::BI__builtin_arm_crc32d:
    CRCIntrinsicID = Intrinsic::arm_crc32w; break;
  case clang::ARM::BI__builtin_arm_crc32cw:
  case clang::ARM::BI__builtin_arm_crc32cd:
    CRCIntrinsicID = Intrinsic::arm_crc32cw; break;
  }

  if (CRCIntrinsicID != Intrinsic::not_intrinsic) {
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    Value *Arg1 = EmitScalarExpr(E->getArg(1));

    // crc32{c,}d intrinsics are implemented as two calls to crc32{c,}w
    // intrinsics, hence we need different codegen for these cases.
    if (BuiltinID == clang::ARM::BI__builtin_arm_crc32d ||
        BuiltinID == clang::ARM::BI__builtin_arm_crc32cd) {
      Value *C1 = llvm::ConstantInt::get(Int64Ty, 32);
      Value *Arg1a = Builder.CreateTruncOrBitCast(Arg1, Int32Ty);
      Value *Arg1b = Builder.CreateLShr(Arg1, C1);
      Arg1b = Builder.CreateTruncOrBitCast(Arg1b, Int32Ty);

      Function *F = CGM.getIntrinsic(CRCIntrinsicID);
      Value *Res = Builder.CreateCall(F, {Arg0, Arg1a});
      return Builder.CreateCall(F, {Res, Arg1b});
    } else {
      Arg1 = Builder.CreateZExtOrBitCast(Arg1, Int32Ty);

      Function *F = CGM.getIntrinsic(CRCIntrinsicID);
      return Builder.CreateCall(F, {Arg0, Arg1});
    }
  }

  if (BuiltinID == clang::ARM::BI__builtin_arm_rsr ||
      BuiltinID == clang::ARM::BI__builtin_arm_rsr64 ||
      BuiltinID == clang::ARM::BI__builtin_arm_rsrp ||
      BuiltinID == clang::ARM::BI__builtin_arm_wsr ||
      BuiltinID == clang::ARM::BI__builtin_arm_wsr64 ||
      BuiltinID == clang::ARM::BI__builtin_arm_wsrp) {

    SpecialRegisterAccessKind AccessKind = Write;
    if (BuiltinID == clang::ARM::BI__builtin_arm_rsr ||
        BuiltinID == clang::ARM::BI__builtin_arm_rsr64 ||
        BuiltinID == clang::ARM::BI__builtin_arm_rsrp)
      AccessKind = VolatileRead;

    bool IsPointerBuiltin = BuiltinID == clang::ARM::BI__builtin_arm_rsrp ||
                            BuiltinID == clang::ARM::BI__builtin_arm_wsrp;

    bool Is64Bit = BuiltinID == clang::ARM::BI__builtin_arm_rsr64 ||
                   BuiltinID == clang::ARM::BI__builtin_arm_wsr64;

    llvm::Type *ValueType;
    llvm::Type *RegisterType;
    if (IsPointerBuiltin) {
      ValueType = VoidPtrTy;
      RegisterType = Int32Ty;
    } else if (Is64Bit) {
      ValueType = RegisterType = Int64Ty;
    } else {
      ValueType = RegisterType = Int32Ty;
    }

    return EmitSpecialRegisterBuiltin(*this, E, RegisterType, ValueType,
                                      AccessKind);
  }

  if (BuiltinID == ARM::BI__builtin_sponentry) {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::sponentry, AllocaInt8PtrTy);
    return Builder.CreateCall(F);
  }

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<MSVCIntrin> MsvcIntId = translateArmToMsvcIntrin(BuiltinID))
    return EmitMSVCBuiltinExpr(*MsvcIntId, E);

  // Deal with MVE builtins
  if (Value *Result = EmitARMMVEBuiltinExpr(BuiltinID, E, ReturnValue, Arch))
    return Result;
  // Handle CDE builtins
  if (Value *Result = EmitARMCDEBuiltinExpr(BuiltinID, E, ReturnValue, Arch))
    return Result;

  // Some intrinsics are equivalent - if they are use the base intrinsic ID.
  auto It = llvm::find_if(NEONEquivalentIntrinsicMap, [BuiltinID](auto &P) {
    return P.first == BuiltinID;
  });
  if (It != end(NEONEquivalentIntrinsicMap))
    BuiltinID = It->second;

  // Find out if any arguments are required to be integer constant
  // expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  auto getAlignmentValue32 = [&](Address addr) -> Value* {
    return Builder.getInt32(addr.getAlignment().getQuantity());
  };

  Address PtrOp0 = Address::invalid();
  Address PtrOp1 = Address::invalid();
  SmallVector<Value*, 4> Ops;
  bool HasExtraArg = HasExtraNeonArgument(BuiltinID);
  unsigned NumArgs = E->getNumArgs() - (HasExtraArg ? 1 : 0);
  for (unsigned i = 0, e = NumArgs; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld1_v:
      case NEON::BI__builtin_neon_vld1q_v:
      case NEON::BI__builtin_neon_vld1q_lane_v:
      case NEON::BI__builtin_neon_vld1_lane_v:
      case NEON::BI__builtin_neon_vld1_dup_v:
      case NEON::BI__builtin_neon_vld1q_dup_v:
      case NEON::BI__builtin_neon_vst1_v:
      case NEON::BI__builtin_neon_vst1q_v:
      case NEON::BI__builtin_neon_vst1q_lane_v:
      case NEON::BI__builtin_neon_vst1_lane_v:
      case NEON::BI__builtin_neon_vst2_v:
      case NEON::BI__builtin_neon_vst2q_v:
      case NEON::BI__builtin_neon_vst2_lane_v:
      case NEON::BI__builtin_neon_vst2q_lane_v:
      case NEON::BI__builtin_neon_vst3_v:
      case NEON::BI__builtin_neon_vst3q_v:
      case NEON::BI__builtin_neon_vst3_lane_v:
      case NEON::BI__builtin_neon_vst3q_lane_v:
      case NEON::BI__builtin_neon_vst4_v:
      case NEON::BI__builtin_neon_vst4q_v:
      case NEON::BI__builtin_neon_vst4_lane_v:
      case NEON::BI__builtin_neon_vst4q_lane_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        PtrOp0 = EmitPointerWithAlignment(E->getArg(0));
        Ops.push_back(PtrOp0.emitRawPointer(*this));
        continue;
      }
    }
    if (i == 1) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld2_v:
      case NEON::BI__builtin_neon_vld2q_v:
      case NEON::BI__builtin_neon_vld3_v:
      case NEON::BI__builtin_neon_vld3q_v:
      case NEON::BI__builtin_neon_vld4_v:
      case NEON::BI__builtin_neon_vld4q_v:
      case NEON::BI__builtin_neon_vld2_lane_v:
      case NEON::BI__builtin_neon_vld2q_lane_v:
      case NEON::BI__builtin_neon_vld3_lane_v:
      case NEON::BI__builtin_neon_vld3q_lane_v:
      case NEON::BI__builtin_neon_vld4_lane_v:
      case NEON::BI__builtin_neon_vld4q_lane_v:
      case NEON::BI__builtin_neon_vld2_dup_v:
      case NEON::BI__builtin_neon_vld2q_dup_v:
      case NEON::BI__builtin_neon_vld3_dup_v:
      case NEON::BI__builtin_neon_vld3q_dup_v:
      case NEON::BI__builtin_neon_vld4_dup_v:
      case NEON::BI__builtin_neon_vld4q_dup_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        PtrOp1 = EmitPointerWithAlignment(E->getArg(1));
        Ops.push_back(PtrOp1.emitRawPointer(*this));
        continue;
      }
    }

    Ops.push_back(EmitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  switch (BuiltinID) {
  default: break;

  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
    return Builder.CreateExtractElement(Ops[0], Ops[1], "vget_lane");

  case NEON::BI__builtin_neon_vrndns_f32: {
    Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *Tys[] = {Arg->getType()};
    Function *F = CGM.getIntrinsic(Intrinsic::roundeven, Tys);
    return Builder.CreateCall(F, {Arg}, "vrndn"); }

  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");

  case NEON::BI__builtin_neon_vsha1h_u32:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1h), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1cq_u32:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1c), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1pq_u32:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1p), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1mq_u32:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1m), Ops,
                        "vsha1h");

  case NEON::BI__builtin_neon_vcvth_bf16_f32: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vcvtbfp2bf), Ops,
                        "vcvtbfp2bf");
  }

  // The ARM _MoveToCoprocessor builtins put the input register value as
  // the first argument, but the LLVM intrinsic expects it as the third one.
  case clang::ARM::BI_MoveToCoprocessor:
  case clang::ARM::BI_MoveToCoprocessor2: {
    Function *F = CGM.getIntrinsic(BuiltinID == clang::ARM::BI_MoveToCoprocessor
                                       ? Intrinsic::arm_mcr
                                       : Intrinsic::arm_mcr2);
    return Builder.CreateCall(F, {Ops[1], Ops[2], Ops[0],
                                  Ops[3], Ops[4], Ops[5]});
  }
  }

  // Get the last argument, which specifies the vector type.
  assert(HasExtraArg);
  const Expr *Arg = E->getArg(E->getNumArgs()-1);
  std::optional<llvm::APSInt> Result =
      Arg->getIntegerConstantExpr(getContext());
  if (!Result)
    return nullptr;

  if (BuiltinID == clang::ARM::BI__builtin_arm_vcvtr_f ||
      BuiltinID == clang::ARM::BI__builtin_arm_vcvtr_d) {
    // Determine the overloaded type of this builtin.
    llvm::Type *Ty;
    if (BuiltinID == clang::ARM::BI__builtin_arm_vcvtr_f)
      Ty = FloatTy;
    else
      Ty = DoubleTy;

    // Determine whether this is an unsigned conversion or not.
    bool usgn = Result->getZExtValue() == 1;
    unsigned Int = usgn ? Intrinsic::arm_vcvtru : Intrinsic::arm_vcvtr;

    // Call the appropriate intrinsic.
    Function *F = CGM.getIntrinsic(Int, Ty);
    return Builder.CreateCall(F, Ops, "vcvtr");
  }

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type = Result->getZExtValue();
  bool usgn = Type.isUnsigned();
  bool rightShift = false;

  llvm::FixedVectorType *VTy =
      GetNeonType(this, Type, getTarget().hasFastHalfType(), false,
                  getTarget().hasBFloat16Type());
  llvm::Type *Ty = VTy;
  if (!Ty)
    return nullptr;

  // Many NEON builtins have identical semantics and uses in ARM and
  // AArch64. Emit these in a single function.
  auto IntrinsicMap = ArrayRef(ARMSIMDIntrinsicMap);
  const ARMVectorIntrinsicInfo *Builtin = findARMVectorIntrinsicInMap(
      IntrinsicMap, BuiltinID, NEONSIMDIntrinsicsProvenSorted);
  if (Builtin)
    return EmitCommonNeonBuiltinExpr(
        Builtin->BuiltinID, Builtin->LLVMIntrinsic, Builtin->AltLLVMIntrinsic,
        Builtin->NameHint, Builtin->TypeModifier, E, Ops, PtrOp0, PtrOp1, Arch);

  unsigned Int;
  switch (BuiltinID) {
  default: return nullptr;
  case NEON::BI__builtin_neon_vld1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use shuffles of
    // one-element vectors to avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      // Extract the other lane.
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      int Lane = cast<ConstantInt>(Ops[2])->getZExtValue();
      Value *SV = llvm::ConstantVector::get(ConstantInt::get(Int32Ty, 1-Lane));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      // Load the value as a one-element vector.
      Ty = llvm::FixedVectorType::get(VTy->getElementType(), 1);
      llvm::Type *Tys[] = {Ty, Int8PtrTy};
      Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld1, Tys);
      Value *Align = getAlignmentValue32(PtrOp0);
      Value *Ld = Builder.CreateCall(F, {Ops[0], Align});
      // Combine them.
      int Indices[] = {1 - Lane, Lane};
      return Builder.CreateShuffleVector(Ops[1], Ld, Indices, "vld1q_lane");
    }
    [[fallthrough]];
  case NEON::BI__builtin_neon_vld1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    PtrOp0 = PtrOp0.withElementType(VTy->getElementType());
    Value *Ld = Builder.CreateLoad(PtrOp0);
    return Builder.CreateInsertElement(Ops[1], Ld, Ops[2], "vld1_lane");
  }
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    Int =
      usgn ? Intrinsic::arm_neon_vqrshiftnu : Intrinsic::arm_neon_vqrshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrn_n",
                        1, true);
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqrshiftnsu, Ty),
                        Ops, "vqrshrun_n", 1, true);
  case NEON::BI__builtin_neon_vqshrn_n_v:
    Int = usgn ? Intrinsic::arm_neon_vqshiftnu : Intrinsic::arm_neon_vqshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrn_n",
                        1, true);
  case NEON::BI__builtin_neon_vqshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqshiftnsu, Ty),
                        Ops, "vqshrun_n", 1, true);
  case NEON::BI__builtin_neon_vrecpe_v:
  case NEON::BI__builtin_neon_vrecpeq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrecpe, Ty),
                        Ops, "vrecpe");
  case NEON::BI__builtin_neon_vrshrn_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrshiftn, Ty),
                        Ops, "vrshrn_n", 1, true);
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, true);
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    Ops[1] = Builder.CreateCall(CGM.getIntrinsic(Int, Ty), {Ops[1], Ops[2]});
    return Builder.CreateAdd(Ops[0], Ops[1], "vrsra_n");
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v:
    rightShift = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v:
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, rightShift);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vshiftins, Ty),
                        Ops, "vsli_n");
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = EmitNeonRShiftImm(Ops[1], Ops[2], Ty, usgn, "vsra_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  case NEON::BI__builtin_neon_vst1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use a shuffle to get
    // a one-element vector and avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      Value *SV = llvm::ConstantVector::get(cast<llvm::Constant>(Ops[2]));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      Ops[2] = getAlignmentValue32(PtrOp0);
      llvm::Type *Tys[] = {Int8PtrTy, Ops[1]->getType()};
      return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst1,
                                                 Tys), Ops);
    }
    [[fallthrough]];
  case NEON::BI__builtin_neon_vst1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    return Builder.CreateStore(Ops[1],
                               PtrOp0.withElementType(Ops[1]->getType()));
  }
  case NEON::BI__builtin_neon_vtbl1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl1),
                        Ops, "vtbl1");
  case NEON::BI__builtin_neon_vtbl2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl2),
                        Ops, "vtbl2");
  case NEON::BI__builtin_neon_vtbl3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl3),
                        Ops, "vtbl3");
  case NEON::BI__builtin_neon_vtbl4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl4),
                        Ops, "vtbl4");
  case NEON::BI__builtin_neon_vtbx1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx1),
                        Ops, "vtbx1");
  case NEON::BI__builtin_neon_vtbx2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx2),
                        Ops, "vtbx2");
  case NEON::BI__builtin_neon_vtbx3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx3),
                        Ops, "vtbx3");
  case NEON::BI__builtin_neon_vtbx4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx4),
                        Ops, "vtbx4");
  }
}

template<typename Integer>
static Integer GetIntegerConstantValue(const Expr *E, ASTContext &Context) {
  return E->getIntegerConstantExpr(Context)->getExtValue();
}

static llvm::Value *SignOrZeroExtend(CGBuilderTy &Builder, llvm::Value *V,
                                     llvm::Type *T, bool Unsigned) {
  // Helper function called by Tablegen-constructed ARM MVE builtin codegen,
  // which finds it convenient to specify signed/unsigned as a boolean flag.
  return Unsigned ? Builder.CreateZExt(V, T) : Builder.CreateSExt(V, T);
}

static llvm::Value *MVEImmediateShr(CGBuilderTy &Builder, llvm::Value *V,
                                    uint32_t Shift, bool Unsigned) {
  // MVE helper function for integer shift right. This must handle signed vs
  // unsigned, and also deal specially with the case where the shift count is
  // equal to the lane size. In LLVM IR, an LShr with that parameter would be
  // undefined behavior, but in MVE it's legal, so we must convert it to code
  // that is not undefined in IR.
  unsigned LaneBits = cast<llvm::VectorType>(V->getType())
                          ->getElementType()
                          ->getPrimitiveSizeInBits();
  if (Shift == LaneBits) {
    // An unsigned shift of the full lane size always generates zero, so we can
    // simply emit a zero vector. A signed shift of the full lane size does the
    // same thing as shifting by one bit fewer.
    if (Unsigned)
      return llvm::Constant::getNullValue(V->getType());
    else
      --Shift;
  }
  return Unsigned ? Builder.CreateLShr(V, Shift) : Builder.CreateAShr(V, Shift);
}

static llvm::Value *ARMMVEVectorSplat(CGBuilderTy &Builder, llvm::Value *V) {
  // MVE-specific helper function for a vector splat, which infers the element
  // count of the output vector by knowing that MVE vectors are all 128 bits
  // wide.
  unsigned Elements = 128 / V->getType()->getPrimitiveSizeInBits();
  return Builder.CreateVectorSplat(Elements, V);
}

static llvm::Value *ARMMVEVectorReinterpret(CGBuilderTy &Builder,
                                            CodeGenFunction *CGF,
                                            llvm::Value *V,
                                            llvm::Type *DestType) {
  // Convert one MVE vector type into another by reinterpreting its in-register
  // format.
  //
  // Little-endian, this is identical to a bitcast (which reinterprets the
  // memory format). But big-endian, they're not necessarily the same, because
  // the register and memory formats map to each other differently depending on
  // the lane size.
  //
  // We generate a bitcast whenever we can (if we're little-endian, or if the
  // lane sizes are the same anyway). Otherwise we fall back to an IR intrinsic
  // that performs the different kind of reinterpretation.
  if (CGF->getTarget().isBigEndian() &&
      V->getType()->getScalarSizeInBits() != DestType->getScalarSizeInBits()) {
    return Builder.CreateCall(
        CGF->CGM.getIntrinsic(Intrinsic::arm_mve_vreinterpretq,
                              {DestType, V->getType()}),
        V);
  } else {
    return Builder.CreateBitCast(V, DestType);
  }
}

static llvm::Value *VectorUnzip(CGBuilderTy &Builder, llvm::Value *V, bool Odd) {
  // Make a shufflevector that extracts every other element of a vector (evens
  // or odds, as desired).
  SmallVector<int, 16> Indices;
  unsigned InputElements =
      cast<llvm::FixedVectorType>(V->getType())->getNumElements();
  for (unsigned i = 0; i < InputElements; i += 2)
    Indices.push_back(i + Odd);
  return Builder.CreateShuffleVector(V, Indices);
}

static llvm::Value *VectorZip(CGBuilderTy &Builder, llvm::Value *V0,
                              llvm::Value *V1) {
  // Make a shufflevector that interleaves two vectors element by element.
  assert(V0->getType() == V1->getType() && "Can't zip different vector types");
  SmallVector<int, 16> Indices;
  unsigned InputElements =
      cast<llvm::FixedVectorType>(V0->getType())->getNumElements();
  for (unsigned i = 0; i < InputElements; i++) {
    Indices.push_back(i);
    Indices.push_back(i + InputElements);
  }
  return Builder.CreateShuffleVector(V0, V1, Indices);
}

template<unsigned HighBit, unsigned OtherBits>
static llvm::Value *ARMMVEConstantSplat(CGBuilderTy &Builder, llvm::Type *VT) {
  // MVE-specific helper function to make a vector splat of a constant such as
  // UINT_MAX or INT_MIN, in which all bits below the highest one are equal.
  llvm::Type *T = cast<llvm::VectorType>(VT)->getElementType();
  unsigned LaneBits = T->getPrimitiveSizeInBits();
  uint32_t Value = HighBit << (LaneBits - 1);
  if (OtherBits)
    Value |= (1UL << (LaneBits - 1)) - 1;
  llvm::Value *Lane = llvm::ConstantInt::get(T, Value);
  return ARMMVEVectorSplat(Builder, Lane);
}

static llvm::Value *ARMMVEVectorElementReverse(CGBuilderTy &Builder,
                                               llvm::Value *V,
                                               unsigned ReverseWidth) {
  // MVE-specific helper function which reverses the elements of a
  // vector within every (ReverseWidth)-bit collection of lanes.
  SmallVector<int, 16> Indices;
  unsigned LaneSize = V->getType()->getScalarSizeInBits();
  unsigned Elements = 128 / LaneSize;
  unsigned Mask = ReverseWidth / LaneSize - 1;
  for (unsigned i = 0; i < Elements; i++)
    Indices.push_back(i ^ Mask);
  return Builder.CreateShuffleVector(V, Indices);
}

Value *CodeGenFunction::EmitARMMVEBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E,
                                              ReturnValueSlot ReturnValue,
                                              llvm::Triple::ArchType Arch) {
  enum class CustomCodeGen { VLD24, VST24 } CustomCodeGenType;
  Intrinsic::ID IRIntr;
  unsigned NumVectors;

  // Code autogenerated by Tablegen will handle all the simple builtins.
  switch (BuiltinID) {
    #include "clang/Basic/arm_mve_builtin_cg.inc"

    // If we didn't match an MVE builtin id at all, go back to the
    // main EmitARMBuiltinExpr.
  default:
    return nullptr;
  }

  // Anything that breaks from that switch is an MVE builtin that
  // needs handwritten code to generate.

  switch (CustomCodeGenType) {

  case CustomCodeGen::VLD24: {
    llvm::SmallVector<Value *, 4> Ops;
    llvm::SmallVector<llvm::Type *, 4> Tys;

    auto MvecCType = E->getType();
    auto MvecLType = ConvertType(MvecCType);
    assert(MvecLType->isStructTy() &&
           "Return type for vld[24]q should be a struct");
    assert(MvecLType->getStructNumElements() == 1 &&
           "Return-type struct for vld[24]q should have one element");
    auto MvecLTypeInner = MvecLType->getStructElementType(0);
    assert(MvecLTypeInner->isArrayTy() &&
           "Return-type struct for vld[24]q should contain an array");
    assert(MvecLTypeInner->getArrayNumElements() == NumVectors &&
           "Array member of return-type struct vld[24]q has wrong length");
    auto VecLType = MvecLTypeInner->getArrayElementType();

    Tys.push_back(VecLType);

    auto Addr = E->getArg(0);
    Ops.push_back(EmitScalarExpr(Addr));
    Tys.push_back(ConvertType(Addr->getType()));

    Function *F = CGM.getIntrinsic(IRIntr, ArrayRef(Tys));
    Value *LoadResult = Builder.CreateCall(F, Ops);
    Value *MvecOut = PoisonValue::get(MvecLType);
    for (unsigned i = 0; i < NumVectors; ++i) {
      Value *Vec = Builder.CreateExtractValue(LoadResult, i);
      MvecOut = Builder.CreateInsertValue(MvecOut, Vec, {0, i});
    }

    if (ReturnValue.isNull())
      return MvecOut;
    else
      return Builder.CreateStore(MvecOut, ReturnValue.getAddress());
  }

  case CustomCodeGen::VST24: {
    llvm::SmallVector<Value *, 4> Ops;
    llvm::SmallVector<llvm::Type *, 4> Tys;

    auto Addr = E->getArg(0);
    Ops.push_back(EmitScalarExpr(Addr));
    Tys.push_back(ConvertType(Addr->getType()));

    auto MvecCType = E->getArg(1)->getType();
    auto MvecLType = ConvertType(MvecCType);
    assert(MvecLType->isStructTy() && "Data type for vst2q should be a struct");
    assert(MvecLType->getStructNumElements() == 1 &&
           "Data-type struct for vst2q should have one element");
    auto MvecLTypeInner = MvecLType->getStructElementType(0);
    assert(MvecLTypeInner->isArrayTy() &&
           "Data-type struct for vst2q should contain an array");
    assert(MvecLTypeInner->getArrayNumElements() == NumVectors &&
           "Array member of return-type struct vld[24]q has wrong length");
    auto VecLType = MvecLTypeInner->getArrayElementType();

    Tys.push_back(VecLType);

    AggValueSlot MvecSlot = CreateAggTemp(MvecCType);
    EmitAggExpr(E->getArg(1), MvecSlot);
    auto Mvec = Builder.CreateLoad(MvecSlot.getAddress());
    for (unsigned i = 0; i < NumVectors; i++)
      Ops.push_back(Builder.CreateExtractValue(Mvec, {0, i}));

    Function *F = CGM.getIntrinsic(IRIntr, ArrayRef(Tys));
    Value *ToReturn = nullptr;
    for (unsigned i = 0; i < NumVectors; i++) {
      Ops.push_back(llvm::ConstantInt::get(Int32Ty, i));
      ToReturn = Builder.CreateCall(F, Ops);
      Ops.pop_back();
    }
    return ToReturn;
  }
  }
  llvm_unreachable("unknown custom codegen type.");
}

Value *CodeGenFunction::EmitARMCDEBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E,
                                              ReturnValueSlot ReturnValue,
                                              llvm::Triple::ArchType Arch) {
  switch (BuiltinID) {
  default:
    return nullptr;
#include "clang/Basic/arm_cde_builtin_cg.inc"
  }
}

static Value *EmitAArch64TblBuiltinExpr(CodeGenFunction &CGF, unsigned BuiltinID,
                                      const CallExpr *E,
                                      SmallVectorImpl<Value *> &Ops,
                                      llvm::Triple::ArchType Arch) {
  unsigned int Int = 0;
  const char *s = nullptr;

  switch (BuiltinID) {
  default:
    return nullptr;
  case NEON::BI__builtin_neon_vtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
  case NEON::BI__builtin_neon_vtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v:
  case NEON::BI__builtin_neon_vtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
  case NEON::BI__builtin_neon_vtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    break;
  case NEON::BI__builtin_neon_vtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
  case NEON::BI__builtin_neon_vtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
  case NEON::BI__builtin_neon_vtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
  case NEON::BI__builtin_neon_vtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    break;
  }

  assert(E->getNumArgs() >= 3);

  // Get the last argument, which specifies the vector type.
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  std::optional<llvm::APSInt> Result =
      Arg->getIntegerConstantExpr(CGF.getContext());
  if (!Result)
    return nullptr;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type = Result->getZExtValue();
  llvm::FixedVectorType *Ty = GetNeonType(&CGF, Type);
  if (!Ty)
    return nullptr;

  CodeGen::CGBuilderTy &Builder = CGF.Builder;

  // AArch64 scalar builtins are not overloaded, they do not have an extra
  // argument that specifies the vector type, need to handle each case.
  switch (BuiltinID) {
  case NEON::BI__builtin_neon_vtbl1_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(0, 1), nullptr, Ops[1],
                              Ty, Intrinsic::aarch64_neon_tbl1, "vtbl1");
  }
  case NEON::BI__builtin_neon_vtbl2_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(0, 2), nullptr, Ops[2],
                              Ty, Intrinsic::aarch64_neon_tbl1, "vtbl1");
  }
  case NEON::BI__builtin_neon_vtbl3_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(0, 3), nullptr, Ops[3],
                              Ty, Intrinsic::aarch64_neon_tbl2, "vtbl2");
  }
  case NEON::BI__builtin_neon_vtbl4_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(0, 4), nullptr, Ops[4],
                              Ty, Intrinsic::aarch64_neon_tbl2, "vtbl2");
  }
  case NEON::BI__builtin_neon_vtbx1_v: {
    Value *TblRes =
        packTBLDVectorList(CGF, ArrayRef(Ops).slice(1, 1), nullptr, Ops[2], Ty,
                           Intrinsic::aarch64_neon_tbl1, "vtbl1");

    llvm::Constant *EightV = ConstantInt::get(Ty, 8);
    Value *CmpRes = Builder.CreateICmp(ICmpInst::ICMP_UGE, Ops[2], EightV);
    CmpRes = Builder.CreateSExt(CmpRes, Ty);

    Value *EltsFromInput = Builder.CreateAnd(CmpRes, Ops[0]);
    Value *EltsFromTbl = Builder.CreateAnd(Builder.CreateNot(CmpRes), TblRes);
    return Builder.CreateOr(EltsFromInput, EltsFromTbl, "vtbx");
  }
  case NEON::BI__builtin_neon_vtbx2_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(1, 2), Ops[0], Ops[3],
                              Ty, Intrinsic::aarch64_neon_tbx1, "vtbx1");
  }
  case NEON::BI__builtin_neon_vtbx3_v: {
    Value *TblRes =
        packTBLDVectorList(CGF, ArrayRef(Ops).slice(1, 3), nullptr, Ops[4], Ty,
                           Intrinsic::aarch64_neon_tbl2, "vtbl2");

    llvm::Constant *TwentyFourV = ConstantInt::get(Ty, 24);
    Value *CmpRes = Builder.CreateICmp(ICmpInst::ICMP_UGE, Ops[4],
                                           TwentyFourV);
    CmpRes = Builder.CreateSExt(CmpRes, Ty);

    Value *EltsFromInput = Builder.CreateAnd(CmpRes, Ops[0]);
    Value *EltsFromTbl = Builder.CreateAnd(Builder.CreateNot(CmpRes), TblRes);
    return Builder.CreateOr(EltsFromInput, EltsFromTbl, "vtbx");
  }
  case NEON::BI__builtin_neon_vtbx4_v: {
    return packTBLDVectorList(CGF, ArrayRef(Ops).slice(1, 4), Ops[0], Ops[5],
                              Ty, Intrinsic::aarch64_neon_tbx2, "vtbx2");
  }
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
    Int = Intrinsic::aarch64_neon_tbl1; s = "vtbl1"; break;
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    Int = Intrinsic::aarch64_neon_tbl2; s = "vtbl2"; break;
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
    Int = Intrinsic::aarch64_neon_tbl3; s = "vtbl3"; break;
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    Int = Intrinsic::aarch64_neon_tbl4; s = "vtbl4"; break;
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
    Int = Intrinsic::aarch64_neon_tbx1; s = "vtbx1"; break;
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
    Int = Intrinsic::aarch64_neon_tbx2; s = "vtbx2"; break;
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
    Int = Intrinsic::aarch64_neon_tbx3; s = "vtbx3"; break;
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    Int = Intrinsic::aarch64_neon_tbx4; s = "vtbx4"; break;
  }
  }

  if (!Int)
    return nullptr;

  Function *F = CGF.CGM.getIntrinsic(Int, Ty);
  return CGF.EmitNeonCall(F, Ops, s);
}

Value *CodeGenFunction::vectorWrapScalar16(Value *Op) {
  auto *VTy = llvm::FixedVectorType::get(Int16Ty, 4);
  Op = Builder.CreateBitCast(Op, Int16Ty);
  Value *V = PoisonValue::get(VTy);
  llvm::Constant *CI = ConstantInt::get(SizeTy, 0);
  Op = Builder.CreateInsertElement(V, Op, CI);
  return Op;
}

/// SVEBuiltinMemEltTy - Returns the memory element type for this memory
/// access builtin.  Only required if it can't be inferred from the base pointer
/// operand.
llvm::Type *CodeGenFunction::SVEBuiltinMemEltTy(const SVETypeFlags &TypeFlags) {
  switch (TypeFlags.getMemEltType()) {
  case SVETypeFlags::MemEltTyDefault:
    return getEltType(TypeFlags);
  case SVETypeFlags::MemEltTyInt8:
    return Builder.getInt8Ty();
  case SVETypeFlags::MemEltTyInt16:
    return Builder.getInt16Ty();
  case SVETypeFlags::MemEltTyInt32:
    return Builder.getInt32Ty();
  case SVETypeFlags::MemEltTyInt64:
    return Builder.getInt64Ty();
  }
  llvm_unreachable("Unknown MemEltType");
}

llvm::Type *CodeGenFunction::getEltType(const SVETypeFlags &TypeFlags) {
  switch (TypeFlags.getEltType()) {
  default:
    llvm_unreachable("Invalid SVETypeFlag!");

  case SVETypeFlags::EltTyMFloat8:
  case SVETypeFlags::EltTyInt8:
    return Builder.getInt8Ty();
  case SVETypeFlags::EltTyInt16:
    return Builder.getInt16Ty();
  case SVETypeFlags::EltTyInt32:
    return Builder.getInt32Ty();
  case SVETypeFlags::EltTyInt64:
    return Builder.getInt64Ty();
  case SVETypeFlags::EltTyInt128:
    return Builder.getInt128Ty();

  case SVETypeFlags::EltTyFloat16:
    return Builder.getHalfTy();
  case SVETypeFlags::EltTyFloat32:
    return Builder.getFloatTy();
  case SVETypeFlags::EltTyFloat64:
    return Builder.getDoubleTy();

  case SVETypeFlags::EltTyBFloat16:
    return Builder.getBFloatTy();

  case SVETypeFlags::EltTyBool8:
  case SVETypeFlags::EltTyBool16:
  case SVETypeFlags::EltTyBool32:
  case SVETypeFlags::EltTyBool64:
    return Builder.getInt1Ty();
  }
}

// Return the llvm predicate vector type corresponding to the specified element
// TypeFlags.
llvm::ScalableVectorType *
CodeGenFunction::getSVEPredType(const SVETypeFlags &TypeFlags) {
  switch (TypeFlags.getEltType()) {
  default: llvm_unreachable("Unhandled SVETypeFlag!");

  case SVETypeFlags::EltTyInt8:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 16);
  case SVETypeFlags::EltTyInt16:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 8);
  case SVETypeFlags::EltTyInt32:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 4);
  case SVETypeFlags::EltTyInt64:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 2);

  case SVETypeFlags::EltTyBFloat16:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 8);
  case SVETypeFlags::EltTyFloat16:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 8);
  case SVETypeFlags::EltTyFloat32:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 4);
  case SVETypeFlags::EltTyFloat64:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 2);

  case SVETypeFlags::EltTyBool8:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 16);
  case SVETypeFlags::EltTyBool16:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 8);
  case SVETypeFlags::EltTyBool32:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 4);
  case SVETypeFlags::EltTyBool64:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 2);
  }
}

// Return the llvm vector type corresponding to the specified element TypeFlags.
llvm::ScalableVectorType *
CodeGenFunction::getSVEType(const SVETypeFlags &TypeFlags) {
  switch (TypeFlags.getEltType()) {
  default:
    llvm_unreachable("Invalid SVETypeFlag!");

  case SVETypeFlags::EltTyInt8:
    return llvm::ScalableVectorType::get(Builder.getInt8Ty(), 16);
  case SVETypeFlags::EltTyInt16:
    return llvm::ScalableVectorType::get(Builder.getInt16Ty(), 8);
  case SVETypeFlags::EltTyInt32:
    return llvm::ScalableVectorType::get(Builder.getInt32Ty(), 4);
  case SVETypeFlags::EltTyInt64:
    return llvm::ScalableVectorType::get(Builder.getInt64Ty(), 2);

  case SVETypeFlags::EltTyMFloat8:
    return llvm::ScalableVectorType::get(Builder.getInt8Ty(), 16);
  case SVETypeFlags::EltTyFloat16:
    return llvm::ScalableVectorType::get(Builder.getHalfTy(), 8);
  case SVETypeFlags::EltTyBFloat16:
    return llvm::ScalableVectorType::get(Builder.getBFloatTy(), 8);
  case SVETypeFlags::EltTyFloat32:
    return llvm::ScalableVectorType::get(Builder.getFloatTy(), 4);
  case SVETypeFlags::EltTyFloat64:
    return llvm::ScalableVectorType::get(Builder.getDoubleTy(), 2);

  case SVETypeFlags::EltTyBool8:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 16);
  case SVETypeFlags::EltTyBool16:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 8);
  case SVETypeFlags::EltTyBool32:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 4);
  case SVETypeFlags::EltTyBool64:
    return llvm::ScalableVectorType::get(Builder.getInt1Ty(), 2);
  }
}

llvm::Value *
CodeGenFunction::EmitSVEAllTruePred(const SVETypeFlags &TypeFlags) {
  Function *Ptrue =
      CGM.getIntrinsic(Intrinsic::aarch64_sve_ptrue, getSVEPredType(TypeFlags));
  return Builder.CreateCall(Ptrue, {Builder.getInt32(/*SV_ALL*/ 31)});
}

constexpr unsigned SVEBitsPerBlock = 128;

static llvm::ScalableVectorType *getSVEVectorForElementType(llvm::Type *EltTy) {
  unsigned NumElts = SVEBitsPerBlock / EltTy->getScalarSizeInBits();
  return llvm::ScalableVectorType::get(EltTy, NumElts);
}

// Reinterpret the input predicate so that it can be used to correctly isolate
// the elements of the specified datatype.
Value *CodeGenFunction::EmitSVEPredicateCast(Value *Pred,
                                             llvm::ScalableVectorType *VTy) {

  if (isa<TargetExtType>(Pred->getType()) &&
      cast<TargetExtType>(Pred->getType())->getName() == "aarch64.svcount")
    return Pred;

  auto *RTy = llvm::VectorType::get(IntegerType::get(getLLVMContext(), 1), VTy);
  if (Pred->getType() == RTy)
    return Pred;

  unsigned IntID;
  llvm::Type *IntrinsicTy;
  switch (VTy->getMinNumElements()) {
  default:
    llvm_unreachable("unsupported element count!");
  case 1:
  case 2:
  case 4:
  case 8:
    IntID = Intrinsic::aarch64_sve_convert_from_svbool;
    IntrinsicTy = RTy;
    break;
  case 16:
    IntID = Intrinsic::aarch64_sve_convert_to_svbool;
    IntrinsicTy = Pred->getType();
    break;
  }

  Function *F = CGM.getIntrinsic(IntID, IntrinsicTy);
  Value *C = Builder.CreateCall(F, Pred);
  assert(C->getType() == RTy && "Unexpected return type!");
  return C;
}

Value *CodeGenFunction::EmitSVEPredicateTupleCast(Value *PredTuple,
                                                  llvm::StructType *Ty) {
  if (PredTuple->getType() == Ty)
    return PredTuple;

  Value *Ret = llvm::PoisonValue::get(Ty);
  for (unsigned I = 0; I < Ty->getNumElements(); ++I) {
    Value *Pred = Builder.CreateExtractValue(PredTuple, I);
    Pred = EmitSVEPredicateCast(
        Pred, cast<llvm::ScalableVectorType>(Ty->getTypeAtIndex(I)));
    Ret = Builder.CreateInsertValue(Ret, Pred, I);
  }

  return Ret;
}

Value *CodeGenFunction::EmitSVEGatherLoad(const SVETypeFlags &TypeFlags,
                                          SmallVectorImpl<Value *> &Ops,
                                          unsigned IntID) {
  auto *ResultTy = getSVEType(TypeFlags);
  auto *OverloadedTy =
      llvm::ScalableVectorType::get(SVEBuiltinMemEltTy(TypeFlags), ResultTy);

  Function *F = nullptr;
  if (Ops[1]->getType()->isVectorTy())
    // This is the "vector base, scalar offset" case. In order to uniquely
    // map this built-in to an LLVM IR intrinsic, we need both the return type
    // and the type of the vector base.
    F = CGM.getIntrinsic(IntID, {OverloadedTy, Ops[1]->getType()});
  else
    // This is the "scalar base, vector offset case". The type of the offset
    // is encoded in the name of the intrinsic. We only need to specify the
    // return type in order to uniquely map this built-in to an LLVM IR
    // intrinsic.
    F = CGM.getIntrinsic(IntID, OverloadedTy);

  // At the ACLE level there's only one predicate type, svbool_t, which is
  // mapped to <n x 16 x i1>. However, this might be incompatible with the
  // actual type being loaded. For example, when loading doubles (i64) the
  // predicate should be <n x 2 x i1> instead. At the IR level the type of
  // the predicate and the data being loaded must match. Cast to the type
  // expected by the intrinsic. The intrinsic itself should be defined in
  // a way than enforces relations between parameter types.
  Ops[0] = EmitSVEPredicateCast(
      Ops[0], cast<llvm::ScalableVectorType>(F->getArg(0)->getType()));

  // Pass 0 when the offset is missing. This can only be applied when using
  // the "vector base" addressing mode for which ACLE allows no offset. The
  // corresponding LLVM IR always requires an offset.
  if (Ops.size() == 2) {
    assert(Ops[1]->getType()->isVectorTy() && "Scalar base requires an offset");
    Ops.push_back(ConstantInt::get(Int64Ty, 0));
  }

  // For "vector base, scalar index" scale the index so that it becomes a
  // scalar offset.
  if (!TypeFlags.isByteIndexed() && Ops[1]->getType()->isVectorTy()) {
    unsigned BytesPerElt =
        OverloadedTy->getElementType()->getScalarSizeInBits() / 8;
    Ops[2] = Builder.CreateShl(Ops[2], Log2_32(BytesPerElt));
  }

  Value *Call = Builder.CreateCall(F, Ops);

  // The following sext/zext is only needed when ResultTy != OverloadedTy. In
  // other cases it's folded into a nop.
  return TypeFlags.isZExtReturn() ? Builder.CreateZExt(Call, ResultTy)
                                  : Builder.CreateSExt(Call, ResultTy);
}

Value *CodeGenFunction::EmitSVEScatterStore(const SVETypeFlags &TypeFlags,
                                            SmallVectorImpl<Value *> &Ops,
                                            unsigned IntID) {
  auto *SrcDataTy = getSVEType(TypeFlags);
  auto *OverloadedTy =
      llvm::ScalableVectorType::get(SVEBuiltinMemEltTy(TypeFlags), SrcDataTy);

  // In ACLE the source data is passed in the last argument, whereas in LLVM IR
  // it's the first argument. Move it accordingly.
  Ops.insert(Ops.begin(), Ops.pop_back_val());

  Function *F = nullptr;
  if (Ops[2]->getType()->isVectorTy())
    // This is the "vector base, scalar offset" case. In order to uniquely
    // map this built-in to an LLVM IR intrinsic, we need both the return type
    // and the type of the vector base.
    F = CGM.getIntrinsic(IntID, {OverloadedTy, Ops[2]->getType()});
  else
    // This is the "scalar base, vector offset case". The type of the offset
    // is encoded in the name of the intrinsic. We only need to specify the
    // return type in order to uniquely map this built-in to an LLVM IR
    // intrinsic.
    F = CGM.getIntrinsic(IntID, OverloadedTy);

  // Pass 0 when the offset is missing. This can only be applied when using
  // the "vector base" addressing mode for which ACLE allows no offset. The
  // corresponding LLVM IR always requires an offset.
  if (Ops.size() == 3) {
    assert(Ops[1]->getType()->isVectorTy() && "Scalar base requires an offset");
    Ops.push_back(ConstantInt::get(Int64Ty, 0));
  }

  // Truncation is needed when SrcDataTy != OverloadedTy. In other cases it's
  // folded into a nop.
  Ops[0] = Builder.CreateTrunc(Ops[0], OverloadedTy);

  // At the ACLE level there's only one predicate type, svbool_t, which is
  // mapped to <n x 16 x i1>. However, this might be incompatible with the
  // actual type being stored. For example, when storing doubles (i64) the
  // predicated should be <n x 2 x i1> instead. At the IR level the type of
  // the predicate and the data being stored must match. Cast to the type
  // expected by the intrinsic. The intrinsic itself should be defined in
  // a way that enforces relations between parameter types.
  Ops[1] = EmitSVEPredicateCast(
      Ops[1], cast<llvm::ScalableVectorType>(F->getArg(1)->getType()));

  // For "vector base, scalar index" scale the index so that it becomes a
  // scalar offset.
  if (!TypeFlags.isByteIndexed() && Ops[2]->getType()->isVectorTy()) {
    unsigned BytesPerElt =
        OverloadedTy->getElementType()->getScalarSizeInBits() / 8;
    Ops[3] = Builder.CreateShl(Ops[3], Log2_32(BytesPerElt));
  }

  return Builder.CreateCall(F, Ops);
}

Value *CodeGenFunction::EmitSVEGatherPrefetch(const SVETypeFlags &TypeFlags,
                                              SmallVectorImpl<Value *> &Ops,
                                              unsigned IntID) {
  // The gather prefetches are overloaded on the vector input - this can either
  // be the vector of base addresses or vector of offsets.
  auto *OverloadedTy = dyn_cast<llvm::ScalableVectorType>(Ops[1]->getType());
  if (!OverloadedTy)
    OverloadedTy = cast<llvm::ScalableVectorType>(Ops[2]->getType());

  // Cast the predicate from svbool_t to the right number of elements.
  Ops[0] = EmitSVEPredicateCast(Ops[0], OverloadedTy);

  // vector + imm addressing modes
  if (Ops[1]->getType()->isVectorTy()) {
    if (Ops.size() == 3) {
      // Pass 0 for 'vector+imm' when the index is omitted.
      Ops.push_back(ConstantInt::get(Int64Ty, 0));

      // The sv_prfop is the last operand in the builtin and IR intrinsic.
      std::swap(Ops[2], Ops[3]);
    } else {
      // Index needs to be passed as scaled offset.
      llvm::Type *MemEltTy = SVEBuiltinMemEltTy(TypeFlags);
      unsigned BytesPerElt = MemEltTy->getPrimitiveSizeInBits() / 8;
      if (BytesPerElt > 1)
        Ops[2] = Builder.CreateShl(Ops[2], Log2_32(BytesPerElt));
    }
  }

  Function *F = CGM.getIntrinsic(IntID, OverloadedTy);
  return Builder.CreateCall(F, Ops);
}

Value *CodeGenFunction::EmitSVEStructLoad(const SVETypeFlags &TypeFlags,
                                          SmallVectorImpl<Value*> &Ops,
                                          unsigned IntID) {
  llvm::ScalableVectorType *VTy = getSVEType(TypeFlags);
  Value *Predicate = EmitSVEPredicateCast(Ops[0], VTy);
  Value *BasePtr = Ops[1];

  // Does the load have an offset?
  if (Ops.size() > 2)
    BasePtr = Builder.CreateGEP(VTy, BasePtr, Ops[2]);

  Function *F = CGM.getIntrinsic(IntID, {VTy});
  return Builder.CreateCall(F, {Predicate, BasePtr});
}

Value *CodeGenFunction::EmitSVEStructStore(const SVETypeFlags &TypeFlags,
                                           SmallVectorImpl<Value*> &Ops,
                                           unsigned IntID) {
  llvm::ScalableVectorType *VTy = getSVEType(TypeFlags);

  unsigned N;
  switch (IntID) {
  case Intrinsic::aarch64_sve_st2:
  case Intrinsic::aarch64_sve_st1_pn_x2:
  case Intrinsic::aarch64_sve_stnt1_pn_x2:
  case Intrinsic::aarch64_sve_st2q:
    N = 2;
    break;
  case Intrinsic::aarch64_sve_st3:
  case Intrinsic::aarch64_sve_st3q:
    N = 3;
    break;
  case Intrinsic::aarch64_sve_st4:
  case Intrinsic::aarch64_sve_st1_pn_x4:
  case Intrinsic::aarch64_sve_stnt1_pn_x4:
  case Intrinsic::aarch64_sve_st4q:
    N = 4;
    break;
  default:
    llvm_unreachable("unknown intrinsic!");
  }

  Value *Predicate = EmitSVEPredicateCast(Ops[0], VTy);
  Value *BasePtr = Ops[1];

  // Does the store have an offset?
  if (Ops.size() > (2 + N))
    BasePtr = Builder.CreateGEP(VTy, BasePtr, Ops[2]);

  // The llvm.aarch64.sve.st2/3/4 intrinsics take legal part vectors, so we
  // need to break up the tuple vector.
  SmallVector<llvm::Value*, 5> Operands;
  for (unsigned I = Ops.size() - N; I < Ops.size(); ++I)
    Operands.push_back(Ops[I]);
  Operands.append({Predicate, BasePtr});
  Function *F = CGM.getIntrinsic(IntID, { VTy });

  return Builder.CreateCall(F, Operands);
}

// SVE2's svpmullb and svpmullt builtins are similar to the svpmullb_pair and
// svpmullt_pair intrinsics, with the exception that their results are bitcast
// to a wider type.
Value *CodeGenFunction::EmitSVEPMull(const SVETypeFlags &TypeFlags,
                                     SmallVectorImpl<Value *> &Ops,
                                     unsigned BuiltinID) {
  // Splat scalar operand to vector (intrinsics with _n infix)
  if (TypeFlags.hasSplatOperand()) {
    unsigned OpNo = TypeFlags.getSplatOperand();
    Ops[OpNo] = EmitSVEDupX(Ops[OpNo]);
  }

  // The pair-wise function has a narrower overloaded type.
  Function *F = CGM.getIntrinsic(BuiltinID, Ops[0]->getType());
  Value *Call = Builder.CreateCall(F, {Ops[0], Ops[1]});

  // Now bitcast to the wider result type.
  llvm::ScalableVectorType *Ty = getSVEType(TypeFlags);
  return EmitSVEReinterpret(Call, Ty);
}

Value *CodeGenFunction::EmitSVEMovl(const SVETypeFlags &TypeFlags,
                                    ArrayRef<Value *> Ops, unsigned BuiltinID) {
  llvm::Type *OverloadedTy = getSVEType(TypeFlags);
  Function *F = CGM.getIntrinsic(BuiltinID, OverloadedTy);
  return Builder.CreateCall(F, {Ops[0], Builder.getInt32(0)});
}

Value *CodeGenFunction::EmitSVEPrefetchLoad(const SVETypeFlags &TypeFlags,
                                            SmallVectorImpl<Value *> &Ops,
                                            unsigned BuiltinID) {
  auto *MemEltTy = SVEBuiltinMemEltTy(TypeFlags);
  auto *VectorTy = getSVEVectorForElementType(MemEltTy);
  auto *MemoryTy = llvm::ScalableVectorType::get(MemEltTy, VectorTy);

  Value *Predicate = EmitSVEPredicateCast(Ops[0], MemoryTy);
  Value *BasePtr = Ops[1];

  // Implement the index operand if not omitted.
  if (Ops.size() > 3)
    BasePtr = Builder.CreateGEP(MemoryTy, BasePtr, Ops[2]);

  Value *PrfOp = Ops.back();

  Function *F = CGM.getIntrinsic(BuiltinID, Predicate->getType());
  return Builder.CreateCall(F, {Predicate, BasePtr, PrfOp});
}

Value *CodeGenFunction::EmitSVEMaskedLoad(const CallExpr *E,
                                          llvm::Type *ReturnTy,
                                          SmallVectorImpl<Value *> &Ops,
                                          unsigned IntrinsicID,
                                          bool IsZExtReturn) {
  QualType LangPTy = E->getArg(1)->getType();
  llvm::Type *MemEltTy = CGM.getTypes().ConvertType(
      LangPTy->castAs<PointerType>()->getPointeeType());

  // Mfloat8 types is stored as a vector, so extra work
  // to extract sclar element type is necessary.
  if (MemEltTy->isVectorTy()) {
    assert(MemEltTy == FixedVectorType::get(Int8Ty, 1) &&
           "Only <1 x i8> expected");
    MemEltTy = cast<llvm::VectorType>(MemEltTy)->getElementType();
  }

  // The vector type that is returned may be different from the
  // eventual type loaded from memory.
  auto VectorTy = cast<llvm::ScalableVectorType>(ReturnTy);
  llvm::ScalableVectorType *MemoryTy = nullptr;
  llvm::ScalableVectorType *PredTy = nullptr;
  bool IsQuadLoad = false;
  switch (IntrinsicID) {
  case Intrinsic::aarch64_sve_ld1uwq:
  case Intrinsic::aarch64_sve_ld1udq:
    MemoryTy = llvm::ScalableVectorType::get(MemEltTy, 1);
    PredTy = llvm::ScalableVectorType::get(
        llvm::Type::getInt1Ty(getLLVMContext()), 1);
    IsQuadLoad = true;
    break;
  default:
    MemoryTy = llvm::ScalableVectorType::get(MemEltTy, VectorTy);
    PredTy = MemoryTy;
    break;
  }

  Value *Predicate = EmitSVEPredicateCast(Ops[0], PredTy);
  Value *BasePtr = Ops[1];

  // Does the load have an offset?
  if (Ops.size() > 2)
    BasePtr = Builder.CreateGEP(MemoryTy, BasePtr, Ops[2]);

  Function *F = CGM.getIntrinsic(IntrinsicID, IsQuadLoad ? VectorTy : MemoryTy);
  auto *Load =
      cast<llvm::Instruction>(Builder.CreateCall(F, {Predicate, BasePtr}));
  auto TBAAInfo = CGM.getTBAAAccessInfo(LangPTy->getPointeeType());
  CGM.DecorateInstructionWithTBAA(Load, TBAAInfo);

  if (IsQuadLoad)
    return Load;

  return IsZExtReturn ? Builder.CreateZExt(Load, VectorTy)
                      : Builder.CreateSExt(Load, VectorTy);
}

Value *CodeGenFunction::EmitSVEMaskedStore(const CallExpr *E,
                                           SmallVectorImpl<Value *> &Ops,
                                           unsigned IntrinsicID) {
  QualType LangPTy = E->getArg(1)->getType();
  llvm::Type *MemEltTy = CGM.getTypes().ConvertType(
      LangPTy->castAs<PointerType>()->getPointeeType());

  // Mfloat8 types is stored as a vector, so extra work
  // to extract sclar element type is necessary.
  if (MemEltTy->isVectorTy()) {
    assert(MemEltTy == FixedVectorType::get(Int8Ty, 1) &&
           "Only <1 x i8> expected");
    MemEltTy = cast<llvm::VectorType>(MemEltTy)->getElementType();
  }

  // The vector type that is stored may be different from the
  // eventual type stored to memory.
  auto VectorTy = cast<llvm::ScalableVectorType>(Ops.back()->getType());
  auto MemoryTy = llvm::ScalableVectorType::get(MemEltTy, VectorTy);

  auto PredTy = MemoryTy;
  auto AddrMemoryTy = MemoryTy;
  bool IsQuadStore = false;

  switch (IntrinsicID) {
  case Intrinsic::aarch64_sve_st1wq:
  case Intrinsic::aarch64_sve_st1dq:
    AddrMemoryTy = llvm::ScalableVectorType::get(MemEltTy, 1);
    PredTy =
        llvm::ScalableVectorType::get(IntegerType::get(getLLVMContext(), 1), 1);
    IsQuadStore = true;
    break;
  default:
    break;
  }
  Value *Predicate = EmitSVEPredicateCast(Ops[0], PredTy);
  Value *BasePtr = Ops[1];

  // Does the store have an offset?
  if (Ops.size() == 4)
    BasePtr = Builder.CreateGEP(AddrMemoryTy, BasePtr, Ops[2]);

  // Last value is always the data
  Value *Val =
      IsQuadStore ? Ops.back() : Builder.CreateTrunc(Ops.back(), MemoryTy);

  Function *F =
      CGM.getIntrinsic(IntrinsicID, IsQuadStore ? VectorTy : MemoryTy);
  auto *Store =
      cast<llvm::Instruction>(Builder.CreateCall(F, {Val, Predicate, BasePtr}));
  auto TBAAInfo = CGM.getTBAAAccessInfo(LangPTy->getPointeeType());
  CGM.DecorateInstructionWithTBAA(Store, TBAAInfo);
  return Store;
}

Value *CodeGenFunction::EmitSMELd1St1(const SVETypeFlags &TypeFlags,
                                      SmallVectorImpl<Value *> &Ops,
                                      unsigned IntID) {
  Ops[2] = EmitSVEPredicateCast(
      Ops[2], getSVEVectorForElementType(SVEBuiltinMemEltTy(TypeFlags)));

  SmallVector<Value *> NewOps;
  NewOps.push_back(Ops[2]);

  llvm::Value *BasePtr = Ops[3];
  llvm::Value *RealSlice = Ops[1];
  // If the intrinsic contains the vnum parameter, multiply it with the vector
  // size in bytes.
  if (Ops.size() == 5) {
    Function *StreamingVectorLength =
        CGM.getIntrinsic(Intrinsic::aarch64_sme_cntsb);
    llvm::Value *StreamingVectorLengthCall =
        Builder.CreateCall(StreamingVectorLength);
    llvm::Value *Mulvl =
        Builder.CreateMul(StreamingVectorLengthCall, Ops[4], "mulvl");
    // The type of the ptr parameter is void *, so use Int8Ty here.
    BasePtr = Builder.CreateGEP(Int8Ty, Ops[3], Mulvl);
    RealSlice = Builder.CreateZExt(RealSlice, Int64Ty);
    RealSlice = Builder.CreateAdd(RealSlice, Ops[4]);
    RealSlice = Builder.CreateTrunc(RealSlice, Int32Ty);
  }
  NewOps.push_back(BasePtr);
  NewOps.push_back(Ops[0]);
  NewOps.push_back(RealSlice);
  Function *F = CGM.getIntrinsic(IntID);
  return Builder.CreateCall(F, NewOps);
}

Value *CodeGenFunction::EmitSMEReadWrite(const SVETypeFlags &TypeFlags,
                                         SmallVectorImpl<Value *> &Ops,
                                         unsigned IntID) {
  auto *VecTy = getSVEType(TypeFlags);
  Function *F = CGM.getIntrinsic(IntID, VecTy);
  if (TypeFlags.isReadZA())
    Ops[1] = EmitSVEPredicateCast(Ops[1], VecTy);
  else if (TypeFlags.isWriteZA())
    Ops[2] = EmitSVEPredicateCast(Ops[2], VecTy);
  return Builder.CreateCall(F, Ops);
}

Value *CodeGenFunction::EmitSMEZero(const SVETypeFlags &TypeFlags,
                                    SmallVectorImpl<Value *> &Ops,
                                    unsigned IntID) {
  // svzero_za() intrinsic zeros the entire za tile and has no paramters.
  if (Ops.size() == 0)
    Ops.push_back(llvm::ConstantInt::get(Int32Ty, 255));
  Function *F = CGM.getIntrinsic(IntID, {});
  return Builder.CreateCall(F, Ops);
}

Value *CodeGenFunction::EmitSMELdrStr(const SVETypeFlags &TypeFlags,
                                      SmallVectorImpl<Value *> &Ops,
                                      unsigned IntID) {
  if (Ops.size() == 2)
    Ops.push_back(Builder.getInt32(0));
  else
    Ops[2] = Builder.CreateIntCast(Ops[2], Int32Ty, true);
  Function *F = CGM.getIntrinsic(IntID, {});
  return Builder.CreateCall(F, Ops);
}

// Limit the usage of scalable llvm IR generated by the ACLE by using the
// sve dup.x intrinsic instead of IRBuilder::CreateVectorSplat.
Value *CodeGenFunction::EmitSVEDupX(Value *Scalar, llvm::Type *Ty) {
  return Builder.CreateVectorSplat(
      cast<llvm::VectorType>(Ty)->getElementCount(), Scalar);
}

Value *CodeGenFunction::EmitSVEDupX(Value *Scalar) {
  if (auto *Ty = Scalar->getType(); Ty->isVectorTy()) {
#ifndef NDEBUG
    auto *VecTy = cast<llvm::VectorType>(Ty);
    ElementCount EC = VecTy->getElementCount();
    assert(EC.isScalar() && VecTy->getElementType() == Int8Ty &&
           "Only <1 x i8> expected");
#endif
    Scalar = Builder.CreateExtractElement(Scalar, uint64_t(0));
  }
  return EmitSVEDupX(Scalar, getSVEVectorForElementType(Scalar->getType()));
}

Value *CodeGenFunction::EmitSVEReinterpret(Value *Val, llvm::Type *Ty) {
  // FIXME: For big endian this needs an additional REV, or needs a separate
  // intrinsic that is code-generated as a no-op, because the LLVM bitcast
  // instruction is defined as 'bitwise' equivalent from memory point of
  // view (when storing/reloading), whereas the svreinterpret builtin
  // implements bitwise equivalent cast from register point of view.
  // LLVM CodeGen for a bitcast must add an explicit REV for big-endian.

  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    Value *Tuple = llvm::PoisonValue::get(Ty);

    for (unsigned I = 0; I < StructTy->getNumElements(); ++I) {
      Value *In = Builder.CreateExtractValue(Val, I);
      Value *Out = Builder.CreateBitCast(In, StructTy->getTypeAtIndex(I));
      Tuple = Builder.CreateInsertValue(Tuple, Out, I);
    }

    return Tuple;
  }

  return Builder.CreateBitCast(Val, Ty);
}

static void InsertExplicitZeroOperand(CGBuilderTy &Builder, llvm::Type *Ty,
                                      SmallVectorImpl<Value *> &Ops) {
  auto *SplatZero = Constant::getNullValue(Ty);
  Ops.insert(Ops.begin(), SplatZero);
}

static void InsertExplicitUndefOperand(CGBuilderTy &Builder, llvm::Type *Ty,
                                       SmallVectorImpl<Value *> &Ops) {
  auto *SplatUndef = UndefValue::get(Ty);
  Ops.insert(Ops.begin(), SplatUndef);
}

SmallVector<llvm::Type *, 2>
CodeGenFunction::getSVEOverloadTypes(const SVETypeFlags &TypeFlags,
                                     llvm::Type *ResultType,
                                     ArrayRef<Value *> Ops) {
  if (TypeFlags.isOverloadNone())
    return {};

  llvm::Type *DefaultType = getSVEType(TypeFlags);

  if (TypeFlags.isOverloadWhileOrMultiVecCvt())
    return {DefaultType, Ops[1]->getType()};

  if (TypeFlags.isOverloadWhileRW())
    return {getSVEPredType(TypeFlags), Ops[0]->getType()};

  if (TypeFlags.isOverloadCvt())
    return {Ops[0]->getType(), Ops.back()->getType()};

  if (TypeFlags.isReductionQV() && !ResultType->isScalableTy() &&
      ResultType->isVectorTy())
    return {ResultType, Ops[1]->getType()};

  assert(TypeFlags.isOverloadDefault() && "Unexpected value for overloads");
  return {DefaultType};
}

Value *CodeGenFunction::EmitSVETupleSetOrGet(const SVETypeFlags &TypeFlags,
                                             ArrayRef<Value *> Ops) {
  assert((TypeFlags.isTupleSet() || TypeFlags.isTupleGet()) &&
         "Expects TypleFlags.isTupleSet() or TypeFlags.isTupleGet()");
  unsigned Idx = cast<ConstantInt>(Ops[1])->getZExtValue();

  if (TypeFlags.isTupleSet())
    return Builder.CreateInsertValue(Ops[0], Ops[2], Idx);
  return Builder.CreateExtractValue(Ops[0], Idx);
}

Value *CodeGenFunction::EmitSVETupleCreate(const SVETypeFlags &TypeFlags,
                                           llvm::Type *Ty,
                                           ArrayRef<Value *> Ops) {
  assert(TypeFlags.isTupleCreate() && "Expects TypleFlag isTupleCreate");

  Value *Tuple = llvm::PoisonValue::get(Ty);
  for (unsigned Idx = 0; Idx < Ops.size(); Idx++)
    Tuple = Builder.CreateInsertValue(Tuple, Ops[Idx], Idx);

  return Tuple;
}

void CodeGenFunction::GetAArch64SVEProcessedOperands(
    unsigned BuiltinID, const CallExpr *E, SmallVectorImpl<Value *> &Ops,
    SVETypeFlags TypeFlags) {
  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  // Tuple set/get only requires one insert/extract vector, which is
  // created by EmitSVETupleSetOrGet.
  bool IsTupleGetOrSet = TypeFlags.isTupleSet() || TypeFlags.isTupleGet();

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    bool IsICE = ICEArguments & (1 << i);
    Value *Arg = EmitScalarExpr(E->getArg(i));

    if (IsICE) {
      // If this is required to be a constant, constant fold it so that we know
      // that the generated intrinsic gets a ConstantInt.
      std::optional<llvm::APSInt> Result =
          E->getArg(i)->getIntegerConstantExpr(getContext());
      assert(Result && "Expected argument to be a constant");

      // Immediates for SVE llvm intrinsics are always 32bit.  We can safely
      // truncate because the immediate has been range checked and no valid
      // immediate requires more than a handful of bits.
      *Result = Result->extOrTrunc(32);
      Ops.push_back(llvm::ConstantInt::get(getLLVMContext(), *Result));
      continue;
    }

    if (isa<StructType>(Arg->getType()) && !IsTupleGetOrSet) {
      for (unsigned I = 0; I < Arg->getType()->getStructNumElements(); ++I)
        Ops.push_back(Builder.CreateExtractValue(Arg, I));

      continue;
    }

    Ops.push_back(Arg);
  }
}

Value *CodeGenFunction::EmitAArch64SVEBuiltinExpr(unsigned BuiltinID,
                                                  const CallExpr *E) {
  llvm::Type *Ty = ConvertType(E->getType());
  if (BuiltinID >= SVE::BI__builtin_sve_reinterpret_s8_s8 &&
      BuiltinID <= SVE::BI__builtin_sve_reinterpret_f64_f64_x4) {
    Value *Val = EmitScalarExpr(E->getArg(0));
    return EmitSVEReinterpret(Val, Ty);
  }

  auto *Builtin = findARMVectorIntrinsicInMap(AArch64SVEIntrinsicMap, BuiltinID,
                                              AArch64SVEIntrinsicsProvenSorted);

  llvm::SmallVector<Value *, 4> Ops;
  SVETypeFlags TypeFlags(Builtin->TypeModifier);
  GetAArch64SVEProcessedOperands(BuiltinID, E, Ops, TypeFlags);

  if (TypeFlags.isLoad())
    return EmitSVEMaskedLoad(E, Ty, Ops, Builtin->LLVMIntrinsic,
                             TypeFlags.isZExtReturn());
  else if (TypeFlags.isStore())
    return EmitSVEMaskedStore(E, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isGatherLoad())
    return EmitSVEGatherLoad(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isScatterStore())
    return EmitSVEScatterStore(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isPrefetch())
    return EmitSVEPrefetchLoad(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isGatherPrefetch())
    return EmitSVEGatherPrefetch(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isStructLoad())
    return EmitSVEStructLoad(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isStructStore())
    return EmitSVEStructStore(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isTupleSet() || TypeFlags.isTupleGet())
    return EmitSVETupleSetOrGet(TypeFlags, Ops);
  else if (TypeFlags.isTupleCreate())
    return EmitSVETupleCreate(TypeFlags, Ty, Ops);
  else if (TypeFlags.isUndef())
    return UndefValue::get(Ty);
  else if (Builtin->LLVMIntrinsic != 0) {
    // Emit set FPMR for intrinsics that require it
    if (TypeFlags.setsFPMR())
      Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_set_fpmr),
                         Ops.pop_back_val());
    if (TypeFlags.getMergeType() == SVETypeFlags::MergeZeroExp)
      InsertExplicitZeroOperand(Builder, Ty, Ops);

    if (TypeFlags.getMergeType() == SVETypeFlags::MergeAnyExp)
      InsertExplicitUndefOperand(Builder, Ty, Ops);

    // Some ACLE builtins leave out the argument to specify the predicate
    // pattern, which is expected to be expanded to an SV_ALL pattern.
    if (TypeFlags.isAppendSVALL())
      Ops.push_back(Builder.getInt32(/*SV_ALL*/ 31));
    if (TypeFlags.isInsertOp1SVALL())
      Ops.insert(&Ops[1], Builder.getInt32(/*SV_ALL*/ 31));

    // Predicates must match the main datatype.
    for (Value *&Op : Ops)
      if (auto PredTy = dyn_cast<llvm::VectorType>(Op->getType()))
        if (PredTy->getElementType()->isIntegerTy(1))
          Op = EmitSVEPredicateCast(Op, getSVEType(TypeFlags));

    // Splat scalar operand to vector (intrinsics with _n infix)
    if (TypeFlags.hasSplatOperand()) {
      unsigned OpNo = TypeFlags.getSplatOperand();
      Ops[OpNo] = EmitSVEDupX(Ops[OpNo]);
    }

    if (TypeFlags.isReverseCompare())
      std::swap(Ops[1], Ops[2]);
    else if (TypeFlags.isReverseUSDOT())
      std::swap(Ops[1], Ops[2]);
    else if (TypeFlags.isReverseMergeAnyBinOp() &&
             TypeFlags.getMergeType() == SVETypeFlags::MergeAny)
      std::swap(Ops[1], Ops[2]);
    else if (TypeFlags.isReverseMergeAnyAccOp() &&
             TypeFlags.getMergeType() == SVETypeFlags::MergeAny)
      std::swap(Ops[1], Ops[3]);

    // Predicated intrinsics with _z suffix need a select w/ zeroinitializer.
    if (TypeFlags.getMergeType() == SVETypeFlags::MergeZero) {
      llvm::Type *OpndTy = Ops[1]->getType();
      auto *SplatZero = Constant::getNullValue(OpndTy);
      Ops[1] = Builder.CreateSelect(Ops[0], Ops[1], SplatZero);
    }

    Function *F = CGM.getIntrinsic(Builtin->LLVMIntrinsic,
                                   getSVEOverloadTypes(TypeFlags, Ty, Ops));
    Value *Call = Builder.CreateCall(F, Ops);

    if (Call->getType() == Ty)
      return Call;

    // Predicate results must be converted to svbool_t.
    if (auto PredTy = dyn_cast<llvm::ScalableVectorType>(Ty))
      return EmitSVEPredicateCast(Call, PredTy);
    if (auto PredTupleTy = dyn_cast<llvm::StructType>(Ty))
      return EmitSVEPredicateTupleCast(Call, PredTupleTy);

    llvm_unreachable("unsupported element count!");
  }

  switch (BuiltinID) {
  default:
    return nullptr;

  case SVE::BI__builtin_sve_svreinterpret_b: {
    auto SVCountTy =
        llvm::TargetExtType::get(getLLVMContext(), "aarch64.svcount");
    Function *CastFromSVCountF =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_convert_to_svbool, SVCountTy);
    return Builder.CreateCall(CastFromSVCountF, Ops[0]);
  }
  case SVE::BI__builtin_sve_svreinterpret_c: {
    auto SVCountTy =
        llvm::TargetExtType::get(getLLVMContext(), "aarch64.svcount");
    Function *CastToSVCountF =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_convert_from_svbool, SVCountTy);
    return Builder.CreateCall(CastToSVCountF, Ops[0]);
  }

  case SVE::BI__builtin_sve_svpsel_lane_b8:
  case SVE::BI__builtin_sve_svpsel_lane_b16:
  case SVE::BI__builtin_sve_svpsel_lane_b32:
  case SVE::BI__builtin_sve_svpsel_lane_b64:
  case SVE::BI__builtin_sve_svpsel_lane_c8:
  case SVE::BI__builtin_sve_svpsel_lane_c16:
  case SVE::BI__builtin_sve_svpsel_lane_c32:
  case SVE::BI__builtin_sve_svpsel_lane_c64: {
    bool IsSVCount = isa<TargetExtType>(Ops[0]->getType());
    assert(((!IsSVCount || cast<TargetExtType>(Ops[0]->getType())->getName() ==
                               "aarch64.svcount")) &&
           "Unexpected TargetExtType");
    auto SVCountTy =
        llvm::TargetExtType::get(getLLVMContext(), "aarch64.svcount");
    Function *CastFromSVCountF =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_convert_to_svbool, SVCountTy);
    Function *CastToSVCountF =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_convert_from_svbool, SVCountTy);

    auto OverloadedTy = getSVEType(SVETypeFlags(Builtin->TypeModifier));
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_sve_psel, OverloadedTy);
    llvm::Value *Ops0 =
        IsSVCount ? Builder.CreateCall(CastFromSVCountF, Ops[0]) : Ops[0];
    llvm::Value *Ops1 = EmitSVEPredicateCast(Ops[1], OverloadedTy);
    llvm::Value *PSel = Builder.CreateCall(F, {Ops0, Ops1, Ops[2]});
    return IsSVCount ? Builder.CreateCall(CastToSVCountF, PSel) : PSel;
  }
  case SVE::BI__builtin_sve_svmov_b_z: {
    // svmov_b_z(pg, op) <=> svand_b_z(pg, op, op)
    SVETypeFlags TypeFlags(Builtin->TypeModifier);
    llvm::Type* OverloadedTy = getSVEType(TypeFlags);
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_sve_and_z, OverloadedTy);
    return Builder.CreateCall(F, {Ops[0], Ops[1], Ops[1]});
  }

  case SVE::BI__builtin_sve_svnot_b_z: {
    // svnot_b_z(pg, op) <=> sveor_b_z(pg, op, pg)
    SVETypeFlags TypeFlags(Builtin->TypeModifier);
    llvm::Type* OverloadedTy = getSVEType(TypeFlags);
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_sve_eor_z, OverloadedTy);
    return Builder.CreateCall(F, {Ops[0], Ops[1], Ops[0]});
  }

  case SVE::BI__builtin_sve_svmovlb_u16:
  case SVE::BI__builtin_sve_svmovlb_u32:
  case SVE::BI__builtin_sve_svmovlb_u64:
    return EmitSVEMovl(TypeFlags, Ops, Intrinsic::aarch64_sve_ushllb);

  case SVE::BI__builtin_sve_svmovlb_s16:
  case SVE::BI__builtin_sve_svmovlb_s32:
  case SVE::BI__builtin_sve_svmovlb_s64:
    return EmitSVEMovl(TypeFlags, Ops, Intrinsic::aarch64_sve_sshllb);

  case SVE::BI__builtin_sve_svmovlt_u16:
  case SVE::BI__builtin_sve_svmovlt_u32:
  case SVE::BI__builtin_sve_svmovlt_u64:
    return EmitSVEMovl(TypeFlags, Ops, Intrinsic::aarch64_sve_ushllt);

  case SVE::BI__builtin_sve_svmovlt_s16:
  case SVE::BI__builtin_sve_svmovlt_s32:
  case SVE::BI__builtin_sve_svmovlt_s64:
    return EmitSVEMovl(TypeFlags, Ops, Intrinsic::aarch64_sve_sshllt);

  case SVE::BI__builtin_sve_svpmullt_u16:
  case SVE::BI__builtin_sve_svpmullt_u64:
  case SVE::BI__builtin_sve_svpmullt_n_u16:
  case SVE::BI__builtin_sve_svpmullt_n_u64:
    return EmitSVEPMull(TypeFlags, Ops, Intrinsic::aarch64_sve_pmullt_pair);

  case SVE::BI__builtin_sve_svpmullb_u16:
  case SVE::BI__builtin_sve_svpmullb_u64:
  case SVE::BI__builtin_sve_svpmullb_n_u16:
  case SVE::BI__builtin_sve_svpmullb_n_u64:
    return EmitSVEPMull(TypeFlags, Ops, Intrinsic::aarch64_sve_pmullb_pair);

  case SVE::BI__builtin_sve_svdup_n_b8:
  case SVE::BI__builtin_sve_svdup_n_b16:
  case SVE::BI__builtin_sve_svdup_n_b32:
  case SVE::BI__builtin_sve_svdup_n_b64: {
    Value *CmpNE =
        Builder.CreateICmpNE(Ops[0], Constant::getNullValue(Ops[0]->getType()));
    llvm::ScalableVectorType *OverloadedTy = getSVEType(TypeFlags);
    Value *Dup = EmitSVEDupX(CmpNE, OverloadedTy);
    return EmitSVEPredicateCast(Dup, cast<llvm::ScalableVectorType>(Ty));
  }

  case SVE::BI__builtin_sve_svdupq_n_b8:
  case SVE::BI__builtin_sve_svdupq_n_b16:
  case SVE::BI__builtin_sve_svdupq_n_b32:
  case SVE::BI__builtin_sve_svdupq_n_b64:
  case SVE::BI__builtin_sve_svdupq_n_u8:
  case SVE::BI__builtin_sve_svdupq_n_s8:
  case SVE::BI__builtin_sve_svdupq_n_u64:
  case SVE::BI__builtin_sve_svdupq_n_f64:
  case SVE::BI__builtin_sve_svdupq_n_s64:
  case SVE::BI__builtin_sve_svdupq_n_u16:
  case SVE::BI__builtin_sve_svdupq_n_f16:
  case SVE::BI__builtin_sve_svdupq_n_bf16:
  case SVE::BI__builtin_sve_svdupq_n_s16:
  case SVE::BI__builtin_sve_svdupq_n_u32:
  case SVE::BI__builtin_sve_svdupq_n_f32:
  case SVE::BI__builtin_sve_svdupq_n_s32: {
    // These builtins are implemented by storing each element to an array and using
    // ld1rq to materialize a vector.
    unsigned NumOpnds = Ops.size();

    bool IsBoolTy =
        cast<llvm::VectorType>(Ty)->getElementType()->isIntegerTy(1);

    // For svdupq_n_b* the element type of is an integer of type 128/numelts,
    // so that the compare can use the width that is natural for the expected
    // number of predicate lanes.
    llvm::Type *EltTy = Ops[0]->getType();
    if (IsBoolTy)
      EltTy = IntegerType::get(getLLVMContext(), SVEBitsPerBlock / NumOpnds);

    SmallVector<llvm::Value *, 16> VecOps;
    for (unsigned I = 0; I < NumOpnds; ++I)
        VecOps.push_back(Builder.CreateZExt(Ops[I], EltTy));
    Value *Vec = BuildVector(VecOps);

    llvm::Type *OverloadedTy = getSVEVectorForElementType(EltTy);
    Value *InsertSubVec = Builder.CreateInsertVector(
        OverloadedTy, PoisonValue::get(OverloadedTy), Vec, uint64_t(0));

    Function *F =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_dupq_lane, OverloadedTy);
    Value *DupQLane =
        Builder.CreateCall(F, {InsertSubVec, Builder.getInt64(0)});

    if (!IsBoolTy)
      return DupQLane;

    SVETypeFlags TypeFlags(Builtin->TypeModifier);
    Value *Pred = EmitSVEAllTruePred(TypeFlags);

    // For svdupq_n_b* we need to add an additional 'cmpne' with '0'.
    F = CGM.getIntrinsic(NumOpnds == 2 ? Intrinsic::aarch64_sve_cmpne
                                       : Intrinsic::aarch64_sve_cmpne_wide,
                         OverloadedTy);
    Value *Call = Builder.CreateCall(
        F, {Pred, DupQLane, EmitSVEDupX(Builder.getInt64(0))});
    return EmitSVEPredicateCast(Call, cast<llvm::ScalableVectorType>(Ty));
  }

  case SVE::BI__builtin_sve_svpfalse_b:
    return ConstantInt::getFalse(Ty);

  case SVE::BI__builtin_sve_svpfalse_c: {
    auto SVBoolTy = ScalableVectorType::get(Builder.getInt1Ty(), 16);
    Function *CastToSVCountF =
        CGM.getIntrinsic(Intrinsic::aarch64_sve_convert_from_svbool, Ty);
    return Builder.CreateCall(CastToSVCountF, ConstantInt::getFalse(SVBoolTy));
  }

  case SVE::BI__builtin_sve_svlen_bf16:
  case SVE::BI__builtin_sve_svlen_f16:
  case SVE::BI__builtin_sve_svlen_f32:
  case SVE::BI__builtin_sve_svlen_f64:
  case SVE::BI__builtin_sve_svlen_s8:
  case SVE::BI__builtin_sve_svlen_s16:
  case SVE::BI__builtin_sve_svlen_s32:
  case SVE::BI__builtin_sve_svlen_s64:
  case SVE::BI__builtin_sve_svlen_u8:
  case SVE::BI__builtin_sve_svlen_u16:
  case SVE::BI__builtin_sve_svlen_u32:
  case SVE::BI__builtin_sve_svlen_u64: {
    SVETypeFlags TF(Builtin->TypeModifier);
    return Builder.CreateElementCount(Ty, getSVEType(TF)->getElementCount());
  }

  case SVE::BI__builtin_sve_svtbl2_u8:
  case SVE::BI__builtin_sve_svtbl2_s8:
  case SVE::BI__builtin_sve_svtbl2_u16:
  case SVE::BI__builtin_sve_svtbl2_s16:
  case SVE::BI__builtin_sve_svtbl2_u32:
  case SVE::BI__builtin_sve_svtbl2_s32:
  case SVE::BI__builtin_sve_svtbl2_u64:
  case SVE::BI__builtin_sve_svtbl2_s64:
  case SVE::BI__builtin_sve_svtbl2_f16:
  case SVE::BI__builtin_sve_svtbl2_bf16:
  case SVE::BI__builtin_sve_svtbl2_f32:
  case SVE::BI__builtin_sve_svtbl2_f64: {
    SVETypeFlags TF(Builtin->TypeModifier);
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_sve_tbl2, getSVEType(TF));
    return Builder.CreateCall(F, Ops);
  }

  case SVE::BI__builtin_sve_svset_neonq_s8:
  case SVE::BI__builtin_sve_svset_neonq_s16:
  case SVE::BI__builtin_sve_svset_neonq_s32:
  case SVE::BI__builtin_sve_svset_neonq_s64:
  case SVE::BI__builtin_sve_svset_neonq_u8:
  case SVE::BI__builtin_sve_svset_neonq_u16:
  case SVE::BI__builtin_sve_svset_neonq_u32:
  case SVE::BI__builtin_sve_svset_neonq_u64:
  case SVE::BI__builtin_sve_svset_neonq_f16:
  case SVE::BI__builtin_sve_svset_neonq_f32:
  case SVE::BI__builtin_sve_svset_neonq_f64:
  case SVE::BI__builtin_sve_svset_neonq_bf16: {
    return Builder.CreateInsertVector(Ty, Ops[0], Ops[1], uint64_t(0));
  }

  case SVE::BI__builtin_sve_svget_neonq_s8:
  case SVE::BI__builtin_sve_svget_neonq_s16:
  case SVE::BI__builtin_sve_svget_neonq_s32:
  case SVE::BI__builtin_sve_svget_neonq_s64:
  case SVE::BI__builtin_sve_svget_neonq_u8:
  case SVE::BI__builtin_sve_svget_neonq_u16:
  case SVE::BI__builtin_sve_svget_neonq_u32:
  case SVE::BI__builtin_sve_svget_neonq_u64:
  case SVE::BI__builtin_sve_svget_neonq_f16:
  case SVE::BI__builtin_sve_svget_neonq_f32:
  case SVE::BI__builtin_sve_svget_neonq_f64:
  case SVE::BI__builtin_sve_svget_neonq_bf16: {
    return Builder.CreateExtractVector(Ty, Ops[0], uint64_t(0));
  }

  case SVE::BI__builtin_sve_svdup_neonq_s8:
  case SVE::BI__builtin_sve_svdup_neonq_s16:
  case SVE::BI__builtin_sve_svdup_neonq_s32:
  case SVE::BI__builtin_sve_svdup_neonq_s64:
  case SVE::BI__builtin_sve_svdup_neonq_u8:
  case SVE::BI__builtin_sve_svdup_neonq_u16:
  case SVE::BI__builtin_sve_svdup_neonq_u32:
  case SVE::BI__builtin_sve_svdup_neonq_u64:
  case SVE::BI__builtin_sve_svdup_neonq_f16:
  case SVE::BI__builtin_sve_svdup_neonq_f32:
  case SVE::BI__builtin_sve_svdup_neonq_f64:
  case SVE::BI__builtin_sve_svdup_neonq_bf16: {
    Value *Insert = Builder.CreateInsertVector(Ty, PoisonValue::get(Ty), Ops[0],
                                               uint64_t(0));
    return Builder.CreateIntrinsic(Intrinsic::aarch64_sve_dupq_lane, {Ty},
                                   {Insert, Builder.getInt64(0)});
  }
  }

  /// Should not happen
  return nullptr;
}

static void swapCommutativeSMEOperands(unsigned BuiltinID,
                                       SmallVectorImpl<Value *> &Ops) {
  unsigned MultiVec;
  switch (BuiltinID) {
  default:
    return;
  case SME::BI__builtin_sme_svsumla_za32_s8_vg4x1:
    MultiVec = 1;
    break;
  case SME::BI__builtin_sme_svsumla_za32_s8_vg4x2:
  case SME::BI__builtin_sme_svsudot_za32_s8_vg1x2:
    MultiVec = 2;
    break;
  case SME::BI__builtin_sme_svsudot_za32_s8_vg1x4:
  case SME::BI__builtin_sme_svsumla_za32_s8_vg4x4:
    MultiVec = 4;
    break;
  }

  if (MultiVec > 0)
    for (unsigned I = 0; I < MultiVec; ++I)
      std::swap(Ops[I + 1], Ops[I + 1 + MultiVec]);
}

Value *CodeGenFunction::EmitAArch64SMEBuiltinExpr(unsigned BuiltinID,
                                                  const CallExpr *E) {
  auto *Builtin = findARMVectorIntrinsicInMap(AArch64SMEIntrinsicMap, BuiltinID,
                                              AArch64SMEIntrinsicsProvenSorted);

  llvm::SmallVector<Value *, 4> Ops;
  SVETypeFlags TypeFlags(Builtin->TypeModifier);
  GetAArch64SVEProcessedOperands(BuiltinID, E, Ops, TypeFlags);

  if (TypeFlags.isLoad() || TypeFlags.isStore())
    return EmitSMELd1St1(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (TypeFlags.isReadZA() || TypeFlags.isWriteZA())
    return EmitSMEReadWrite(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (BuiltinID == SME::BI__builtin_sme_svzero_mask_za ||
           BuiltinID == SME::BI__builtin_sme_svzero_za)
    return EmitSMEZero(TypeFlags, Ops, Builtin->LLVMIntrinsic);
  else if (BuiltinID == SME::BI__builtin_sme_svldr_vnum_za ||
           BuiltinID == SME::BI__builtin_sme_svstr_vnum_za ||
           BuiltinID == SME::BI__builtin_sme_svldr_za ||
           BuiltinID == SME::BI__builtin_sme_svstr_za)
    return EmitSMELdrStr(TypeFlags, Ops, Builtin->LLVMIntrinsic);

  // Emit set FPMR for intrinsics that require it
  if (TypeFlags.setsFPMR())
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_set_fpmr),
                       Ops.pop_back_val());
  // Handle builtins which require their multi-vector operands to be swapped
  swapCommutativeSMEOperands(BuiltinID, Ops);

  // Should not happen!
  if (Builtin->LLVMIntrinsic == 0)
    return nullptr;

  // Predicates must match the main datatype.
  for (Value *&Op : Ops)
    if (auto PredTy = dyn_cast<llvm::VectorType>(Op->getType()))
      if (PredTy->getElementType()->isIntegerTy(1))
        Op = EmitSVEPredicateCast(Op, getSVEType(TypeFlags));

  Function *F =
      TypeFlags.isOverloadNone()
          ? CGM.getIntrinsic(Builtin->LLVMIntrinsic)
          : CGM.getIntrinsic(Builtin->LLVMIntrinsic, {getSVEType(TypeFlags)});

  return Builder.CreateCall(F, Ops);
}

/// Helper for the read/write/add/inc X18 builtins: read the X18 register and
/// return it as an i8 pointer.
Value *readX18AsPtr(CodeGenFunction &CGF) {
  LLVMContext &Context = CGF.CGM.getLLVMContext();
  llvm::Metadata *Ops[] = {llvm::MDString::get(Context, "x18")};
  llvm::MDNode *RegName = llvm::MDNode::get(Context, Ops);
  llvm::Value *Metadata = llvm::MetadataAsValue::get(Context, RegName);
  llvm::Function *F =
      CGF.CGM.getIntrinsic(Intrinsic::read_register, {CGF.Int64Ty});
  llvm::Value *X18 = CGF.Builder.CreateCall(F, Metadata);
  return CGF.Builder.CreateIntToPtr(X18, CGF.Int8PtrTy);
}

Value *CodeGenFunction::EmitAArch64BuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E,
                                               llvm::Triple::ArchType Arch) {
  if (BuiltinID >= clang::AArch64::FirstSVEBuiltin &&
      BuiltinID <= clang::AArch64::LastSVEBuiltin)
    return EmitAArch64SVEBuiltinExpr(BuiltinID, E);

  if (BuiltinID >= clang::AArch64::FirstSMEBuiltin &&
      BuiltinID <= clang::AArch64::LastSMEBuiltin)
    return EmitAArch64SMEBuiltinExpr(BuiltinID, E);

  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    return EmitAArch64CpuSupports(E);

  unsigned HintID = static_cast<unsigned>(-1);
  switch (BuiltinID) {
  default: break;
  case clang::AArch64::BI__builtin_arm_nop:
    HintID = 0;
    break;
  case clang::AArch64::BI__builtin_arm_yield:
  case clang::AArch64::BI__yield:
    HintID = 1;
    break;
  case clang::AArch64::BI__builtin_arm_wfe:
  case clang::AArch64::BI__wfe:
    HintID = 2;
    break;
  case clang::AArch64::BI__builtin_arm_wfi:
  case clang::AArch64::BI__wfi:
    HintID = 3;
    break;
  case clang::AArch64::BI__builtin_arm_sev:
  case clang::AArch64::BI__sev:
    HintID = 4;
    break;
  case clang::AArch64::BI__builtin_arm_sevl:
  case clang::AArch64::BI__sevl:
    HintID = 5;
    break;
  }

  if (HintID != static_cast<unsigned>(-1)) {
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_hint);
    return Builder.CreateCall(F, llvm::ConstantInt::get(Int32Ty, HintID));
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_trap) {
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_break);
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(F, Builder.CreateZExt(Arg, CGM.Int32Ty));
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_get_sme_state) {
    // Create call to __arm_sme_state and store the results to the two pointers.
    CallInst *CI = EmitRuntimeCall(CGM.CreateRuntimeFunction(
        llvm::FunctionType::get(StructType::get(CGM.Int64Ty, CGM.Int64Ty), {},
                                false),
        "__arm_sme_state"));
    auto Attrs = AttributeList().addFnAttribute(getLLVMContext(),
                                                "aarch64_pstate_sm_compatible");
    CI->setAttributes(Attrs);
    CI->setCallingConv(
        llvm::CallingConv::
            AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2);
    Builder.CreateStore(Builder.CreateExtractValue(CI, 0),
                        EmitPointerWithAlignment(E->getArg(0)));
    return Builder.CreateStore(Builder.CreateExtractValue(CI, 1),
                               EmitPointerWithAlignment(E->getArg(1)));
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "rbit of unusual size!");
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::bitreverse, Arg->getType()), Arg, "rbit");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit64) {
    assert((getContext().getTypeSize(E->getType()) == 64) &&
           "rbit of unusual size!");
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::bitreverse, Arg->getType()), Arg, "rbit");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clz ||
      BuiltinID == clang::AArch64::BI__builtin_arm_clz64) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Arg->getType());
    Value *Res = Builder.CreateCall(F, {Arg, Builder.getInt1(false)});
    if (BuiltinID == clang::AArch64::BI__builtin_arm_clz64)
      Res = Builder.CreateTrunc(Res, Builder.getInt32Ty());
    return Res;
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_cls), Arg,
                              "cls");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls64) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_cls64), Arg,
                              "cls");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32z) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *Ty = Arg->getType();
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_frint32z, Ty),
                              Arg, "frint32z");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64z) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *Ty = Arg->getType();
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_frint64z, Ty),
                              Arg, "frint64z");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32x) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *Ty = Arg->getType();
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_frint32x, Ty),
                              Arg, "frint32x");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64x) {
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *Ty = Arg->getType();
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::aarch64_frint64x, Ty),
                              Arg, "frint64x");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_jcvt) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "__jcvt of unusual size!");
    llvm::Value *Arg = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::aarch64_fjcvtzs), Arg);
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv0) {
    llvm::Value *MemAddr = EmitScalarExpr(E->getArg(0));
    llvm::Value *ValPtr = EmitScalarExpr(E->getArg(1));

    if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b) {
      // Load from the address via an LLVM intrinsic, receiving a
      // tuple of 8 i64 words, and store each one to ValPtr.
      Function *F = CGM.getIntrinsic(Intrinsic::aarch64_ld64b);
      llvm::Value *Val = Builder.CreateCall(F, MemAddr);
      llvm::Value *ToRet;
      for (size_t i = 0; i < 8; i++) {
        llvm::Value *ValOffsetPtr =
            Builder.CreateGEP(Int64Ty, ValPtr, Builder.getInt32(i));
        Address Addr =
            Address(ValOffsetPtr, Int64Ty, CharUnits::fromQuantity(8));
        ToRet = Builder.CreateStore(Builder.CreateExtractValue(Val, i), Addr);
      }
      return ToRet;
    } else {
      // Load 8 i64 words from ValPtr, and store them to the address
      // via an LLVM intrinsic.
      SmallVector<llvm::Value *, 9> Args;
      Args.push_back(MemAddr);
      for (size_t i = 0; i < 8; i++) {
        llvm::Value *ValOffsetPtr =
            Builder.CreateGEP(Int64Ty, ValPtr, Builder.getInt32(i));
        Address Addr =
            Address(ValOffsetPtr, Int64Ty, CharUnits::fromQuantity(8));
        Args.push_back(Builder.CreateLoad(Addr));
      }

      auto Intr = (BuiltinID == clang::AArch64::BI__builtin_arm_st64b
                       ? Intrinsic::aarch64_st64b
                   : BuiltinID == clang::AArch64::BI__builtin_arm_st64bv
                       ? Intrinsic::aarch64_st64bv
                       : Intrinsic::aarch64_st64bv0);
      Function *F = CGM.getIntrinsic(Intr);
      return Builder.CreateCall(F, Args);
    }
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rndr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rndrrs) {

    auto Intr = (BuiltinID == clang::AArch64::BI__builtin_arm_rndr
                     ? Intrinsic::aarch64_rndr
                     : Intrinsic::aarch64_rndrrs);
    Function *F = CGM.getIntrinsic(Intr);
    llvm::Value *Val = Builder.CreateCall(F);
    Value *RandomValue = Builder.CreateExtractValue(Val, 0);
    Value *Status = Builder.CreateExtractValue(Val, 1);

    Address MemAddress = EmitPointerWithAlignment(E->getArg(0));
    Builder.CreateStore(RandomValue, MemAddress);
    Status = Builder.CreateZExt(Status, Int32Ty);
    return Status;
  }

  if (BuiltinID == clang::AArch64::BI__clear_cache) {
    assert(E->getNumArgs() == 2 && "__clear_cache takes 2 arguments");
    const FunctionDecl *FD = E->getDirectCallee();
    Value *Ops[2];
    for (unsigned i = 0; i < 2; i++)
      Ops[i] = EmitScalarExpr(E->getArg(i));
    llvm::Type *Ty = CGM.getTypes().ConvertType(FD->getType());
    llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    StringRef Name = FD->getName();
    return EmitNounwindRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Ops);
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) &&
      getContext().getTypeSize(E->getType()) == 128) {
    Function *F =
        CGM.getIntrinsic(BuiltinID == clang::AArch64::BI__builtin_arm_ldaex
                             ? Intrinsic::aarch64_ldaxp
                             : Intrinsic::aarch64_ldxp);

    Value *LdPtr = EmitScalarExpr(E->getArg(0));
    Value *Val = Builder.CreateCall(F, LdPtr, "ldxp");

    Value *Val0 = Builder.CreateExtractValue(Val, 1);
    Value *Val1 = Builder.CreateExtractValue(Val, 0);
    llvm::Type *Int128Ty = llvm::IntegerType::get(getLLVMContext(), 128);
    Val0 = Builder.CreateZExt(Val0, Int128Ty);
    Val1 = Builder.CreateZExt(Val1, Int128Ty);

    Value *ShiftCst = llvm::ConstantInt::get(Int128Ty, 64);
    Val = Builder.CreateShl(Val0, ShiftCst, "shl", true /* nuw */);
    Val = Builder.CreateOr(Val, Val1);
    return Builder.CreateBitCast(Val, ConvertType(E->getType()));
  } else if (BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
             BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) {
    Value *LoadAddr = EmitScalarExpr(E->getArg(0));

    QualType Ty = E->getType();
    llvm::Type *RealResTy = ConvertType(Ty);
    llvm::Type *IntTy =
        llvm::IntegerType::get(getLLVMContext(), getContext().getTypeSize(Ty));

    Function *F =
        CGM.getIntrinsic(BuiltinID == clang::AArch64::BI__builtin_arm_ldaex
                             ? Intrinsic::aarch64_ldaxr
                             : Intrinsic::aarch64_ldxr,
                         UnqualPtrTy);
    CallInst *Val = Builder.CreateCall(F, LoadAddr, "ldxr");
    Val->addParamAttr(
        0, Attribute::get(getLLVMContext(), Attribute::ElementType, IntTy));

    if (RealResTy->isPointerTy())
      return Builder.CreateIntToPtr(Val, RealResTy);

    llvm::Type *IntResTy = llvm::IntegerType::get(
        getLLVMContext(), CGM.getDataLayout().getTypeSizeInBits(RealResTy));
    return Builder.CreateBitCast(Builder.CreateTruncOrBitCast(Val, IntResTy),
                                 RealResTy);
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_stlex) &&
      getContext().getTypeSize(E->getArg(0)->getType()) == 128) {
    Function *F =
        CGM.getIntrinsic(BuiltinID == clang::AArch64::BI__builtin_arm_stlex
                             ? Intrinsic::aarch64_stlxp
                             : Intrinsic::aarch64_stxp);
    llvm::Type *STy = llvm::StructType::get(Int64Ty, Int64Ty);

    Address Tmp = CreateMemTemp(E->getArg(0)->getType());
    EmitAnyExprToMem(E->getArg(0), Tmp, Qualifiers(), /*init*/ true);

    Tmp = Tmp.withElementType(STy);
    llvm::Value *Val = Builder.CreateLoad(Tmp);

    Value *Arg0 = Builder.CreateExtractValue(Val, 0);
    Value *Arg1 = Builder.CreateExtractValue(Val, 1);
    Value *StPtr = EmitScalarExpr(E->getArg(1));
    return Builder.CreateCall(F, {Arg0, Arg1, StPtr}, "stxp");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
      BuiltinID == clang::AArch64::BI__builtin_arm_stlex) {
    Value *StoreVal = EmitScalarExpr(E->getArg(0));
    Value *StoreAddr = EmitScalarExpr(E->getArg(1));

    QualType Ty = E->getArg(0)->getType();
    llvm::Type *StoreTy =
        llvm::IntegerType::get(getLLVMContext(), getContext().getTypeSize(Ty));

    if (StoreVal->getType()->isPointerTy())
      StoreVal = Builder.CreatePtrToInt(StoreVal, Int64Ty);
    else {
      llvm::Type *IntTy = llvm::IntegerType::get(
          getLLVMContext(),
          CGM.getDataLayout().getTypeSizeInBits(StoreVal->getType()));
      StoreVal = Builder.CreateBitCast(StoreVal, IntTy);
      StoreVal = Builder.CreateZExtOrBitCast(StoreVal, Int64Ty);
    }

    Function *F =
        CGM.getIntrinsic(BuiltinID == clang::AArch64::BI__builtin_arm_stlex
                             ? Intrinsic::aarch64_stlxr
                             : Intrinsic::aarch64_stxr,
                         StoreAddr->getType());
    CallInst *CI = Builder.CreateCall(F, {StoreVal, StoreAddr}, "stxr");
    CI->addParamAttr(
        1, Attribute::get(getLLVMContext(), Attribute::ElementType, StoreTy));
    return CI;
  }

  if (BuiltinID == clang::AArch64::BI__getReg) {
    Expr::EvalResult Result;
    if (!E->getArg(0)->EvaluateAsInt(Result, CGM.getContext()))
      llvm_unreachable("Sema will ensure that the parameter is constant");

    llvm::APSInt Value = Result.Val.getInt();
    LLVMContext &Context = CGM.getLLVMContext();
    std::string Reg = Value == 31 ? "sp" : "x" + toString(Value, 10);

    llvm::Metadata *Ops[] = {llvm::MDString::get(Context, Reg)};
    llvm::MDNode *RegName = llvm::MDNode::get(Context, Ops);
    llvm::Value *Metadata = llvm::MetadataAsValue::get(Context, RegName);

    llvm::Function *F =
        CGM.getIntrinsic(Intrinsic::read_register, {Int64Ty});
    return Builder.CreateCall(F, Metadata);
  }

  if (BuiltinID == clang::AArch64::BI__break) {
    Expr::EvalResult Result;
    if (!E->getArg(0)->EvaluateAsInt(Result, CGM.getContext()))
      llvm_unreachable("Sema will ensure that the parameter is constant");

    llvm::Function *F = CGM.getIntrinsic(Intrinsic::aarch64_break);
    return Builder.CreateCall(F, {EmitScalarExpr(E->getArg(0))});
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clrex) {
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_clrex);
    return Builder.CreateCall(F);
  }

  if (BuiltinID == clang::AArch64::BI_ReadWriteBarrier)
    return Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                               llvm::SyncScope::SingleThread);

  // CRC32
  Intrinsic::ID CRCIntrinsicID = Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case clang::AArch64::BI__builtin_arm_crc32b:
    CRCIntrinsicID = Intrinsic::aarch64_crc32b; break;
  case clang::AArch64::BI__builtin_arm_crc32cb:
    CRCIntrinsicID = Intrinsic::aarch64_crc32cb; break;
  case clang::AArch64::BI__builtin_arm_crc32h:
    CRCIntrinsicID = Intrinsic::aarch64_crc32h; break;
  case clang::AArch64::BI__builtin_arm_crc32ch:
    CRCIntrinsicID = Intrinsic::aarch64_crc32ch; break;
  case clang::AArch64::BI__builtin_arm_crc32w:
    CRCIntrinsicID = Intrinsic::aarch64_crc32w; break;
  case clang::AArch64::BI__builtin_arm_crc32cw:
    CRCIntrinsicID = Intrinsic::aarch64_crc32cw; break;
  case clang::AArch64::BI__builtin_arm_crc32d:
    CRCIntrinsicID = Intrinsic::aarch64_crc32x; break;
  case clang::AArch64::BI__builtin_arm_crc32cd:
    CRCIntrinsicID = Intrinsic::aarch64_crc32cx; break;
  }

  if (CRCIntrinsicID != Intrinsic::not_intrinsic) {
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    Value *Arg1 = EmitScalarExpr(E->getArg(1));
    Function *F = CGM.getIntrinsic(CRCIntrinsicID);

    llvm::Type *DataTy = F->getFunctionType()->getParamType(1);
    Arg1 = Builder.CreateZExtOrBitCast(Arg1, DataTy);

    return Builder.CreateCall(F, {Arg0, Arg1});
  }

  // Memory Operations (MOPS)
  if (BuiltinID == AArch64::BI__builtin_arm_mops_memset_tag) {
    Value *Dst = EmitScalarExpr(E->getArg(0));
    Value *Val = EmitScalarExpr(E->getArg(1));
    Value *Size = EmitScalarExpr(E->getArg(2));
    Val = Builder.CreateTrunc(Val, Int8Ty);
    Size = Builder.CreateIntCast(Size, Int64Ty, false);
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::aarch64_mops_memset_tag), {Dst, Val, Size});
  }

  // Memory Tagging Extensions (MTE) Intrinsics
  Intrinsic::ID MTEIntrinsicID = Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case clang::AArch64::BI__builtin_arm_irg:
    MTEIntrinsicID = Intrinsic::aarch64_irg; break;
  case clang::AArch64::BI__builtin_arm_addg:
    MTEIntrinsicID = Intrinsic::aarch64_addg; break;
  case clang::AArch64::BI__builtin_arm_gmi:
    MTEIntrinsicID = Intrinsic::aarch64_gmi; break;
  case clang::AArch64::BI__builtin_arm_ldg:
    MTEIntrinsicID = Intrinsic::aarch64_ldg; break;
  case clang::AArch64::BI__builtin_arm_stg:
    MTEIntrinsicID = Intrinsic::aarch64_stg; break;
  case clang::AArch64::BI__builtin_arm_subp:
    MTEIntrinsicID = Intrinsic::aarch64_subp; break;
  }

  if (MTEIntrinsicID != Intrinsic::not_intrinsic) {
    if (MTEIntrinsicID == Intrinsic::aarch64_irg) {
      Value *Pointer = EmitScalarExpr(E->getArg(0));
      Value *Mask = EmitScalarExpr(E->getArg(1));

      Mask = Builder.CreateZExt(Mask, Int64Ty);
      return Builder.CreateCall(CGM.getIntrinsic(MTEIntrinsicID),
                                {Pointer, Mask});
    }
    if (MTEIntrinsicID == Intrinsic::aarch64_addg) {
      Value *Pointer = EmitScalarExpr(E->getArg(0));
      Value *TagOffset = EmitScalarExpr(E->getArg(1));

      TagOffset = Builder.CreateZExt(TagOffset, Int64Ty);
      return Builder.CreateCall(CGM.getIntrinsic(MTEIntrinsicID),
                                {Pointer, TagOffset});
    }
    if (MTEIntrinsicID == Intrinsic::aarch64_gmi) {
      Value *Pointer = EmitScalarExpr(E->getArg(0));
      Value *ExcludedMask = EmitScalarExpr(E->getArg(1));

      ExcludedMask = Builder.CreateZExt(ExcludedMask, Int64Ty);
      return Builder.CreateCall(
                       CGM.getIntrinsic(MTEIntrinsicID), {Pointer, ExcludedMask});
    }
    // Although it is possible to supply a different return
    // address (first arg) to this intrinsic, for now we set
    // return address same as input address.
    if (MTEIntrinsicID == Intrinsic::aarch64_ldg) {
      Value *TagAddress = EmitScalarExpr(E->getArg(0));
      return Builder.CreateCall(CGM.getIntrinsic(MTEIntrinsicID),
                                {TagAddress, TagAddress});
    }
    // Although it is possible to supply a different tag (to set)
    // to this intrinsic (as first arg), for now we supply
    // the tag that is in input address arg (common use case).
    if (MTEIntrinsicID == Intrinsic::aarch64_stg) {
      Value *TagAddress = EmitScalarExpr(E->getArg(0));
      return Builder.CreateCall(CGM.getIntrinsic(MTEIntrinsicID),
                                {TagAddress, TagAddress});
    }
    if (MTEIntrinsicID == Intrinsic::aarch64_subp) {
      Value *PointerA = EmitScalarExpr(E->getArg(0));
      Value *PointerB = EmitScalarExpr(E->getArg(1));
      return Builder.CreateCall(
                       CGM.getIntrinsic(MTEIntrinsicID), {PointerA, PointerB});
    }
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsrp) {

    SpecialRegisterAccessKind AccessKind = Write;
    if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsrp)
      AccessKind = VolatileRead;

    bool IsPointerBuiltin = BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
                            BuiltinID == clang::AArch64::BI__builtin_arm_wsrp;

    bool Is32Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
                   BuiltinID == clang::AArch64::BI__builtin_arm_wsr;

    bool Is128Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
                    BuiltinID == clang::AArch64::BI__builtin_arm_wsr128;

    llvm::Type *ValueType;
    llvm::Type *RegisterType = Int64Ty;
    if (Is32Bit) {
      ValueType = Int32Ty;
    } else if (Is128Bit) {
      llvm::Type *Int128Ty =
          llvm::IntegerType::getInt128Ty(CGM.getLLVMContext());
      ValueType = Int128Ty;
      RegisterType = Int128Ty;
    } else if (IsPointerBuiltin) {
      ValueType = VoidPtrTy;
    } else {
      ValueType = Int64Ty;
    };

    return EmitSpecialRegisterBuiltin(*this, E, RegisterType, ValueType,
                                      AccessKind);
  }

  if (BuiltinID == clang::AArch64::BI_ReadStatusReg ||
      BuiltinID == clang::AArch64::BI_WriteStatusReg ||
      BuiltinID == clang::AArch64::BI__sys) {
    LLVMContext &Context = CGM.getLLVMContext();

    unsigned SysReg =
      E->getArg(0)->EvaluateKnownConstInt(getContext()).getZExtValue();

    std::string SysRegStr;
    unsigned SysRegOp0 = (BuiltinID == clang::AArch64::BI_ReadStatusReg ||
                          BuiltinID == clang::AArch64::BI_WriteStatusReg)
                             ? ((1 << 1) | ((SysReg >> 14) & 1))
                             : 1;
    llvm::raw_string_ostream(SysRegStr)
        << SysRegOp0 << ":" << ((SysReg >> 11) & 7) << ":"
        << ((SysReg >> 7) & 15) << ":" << ((SysReg >> 3) & 15) << ":"
        << (SysReg & 7);

    llvm::Metadata *Ops[] = { llvm::MDString::get(Context, SysRegStr) };
    llvm::MDNode *RegName = llvm::MDNode::get(Context, Ops);
    llvm::Value *Metadata = llvm::MetadataAsValue::get(Context, RegName);

    llvm::Type *RegisterType = Int64Ty;
    llvm::Type *Types[] = { RegisterType };

    if (BuiltinID == clang::AArch64::BI_ReadStatusReg) {
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::read_register, Types);

      return Builder.CreateCall(F, Metadata);
    }

    llvm::Function *F = CGM.getIntrinsic(Intrinsic::write_register, Types);
    llvm::Value *ArgValue = EmitScalarExpr(E->getArg(1));
    llvm::Value *Result = Builder.CreateCall(F, {Metadata, ArgValue});
    if (BuiltinID == clang::AArch64::BI__sys) {
      // Return 0 for convenience, even though MSVC returns some other undefined
      // value.
      Result = ConstantInt::get(Builder.getInt32Ty(), 0);
    }
    return Result;
  }

  if (BuiltinID == clang::AArch64::BI_AddressOfReturnAddress) {
    llvm::Function *F =
        CGM.getIntrinsic(Intrinsic::addressofreturnaddress, AllocaInt8PtrTy);
    return Builder.CreateCall(F);
  }

  if (BuiltinID == clang::AArch64::BI__builtin_sponentry) {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::sponentry, AllocaInt8PtrTy);
    return Builder.CreateCall(F);
  }

  if (BuiltinID == clang::AArch64::BI__mulh ||
      BuiltinID == clang::AArch64::BI__umulh) {
    llvm::Type *ResType = ConvertType(E->getType());
    llvm::Type *Int128Ty = llvm::IntegerType::get(getLLVMContext(), 128);

    bool IsSigned = BuiltinID == clang::AArch64::BI__mulh;
    Value *LHS =
        Builder.CreateIntCast(EmitScalarExpr(E->getArg(0)), Int128Ty, IsSigned);
    Value *RHS =
        Builder.CreateIntCast(EmitScalarExpr(E->getArg(1)), Int128Ty, IsSigned);

    Value *MulResult, *HigherBits;
    if (IsSigned) {
      MulResult = Builder.CreateNSWMul(LHS, RHS);
      HigherBits = Builder.CreateAShr(MulResult, 64);
    } else {
      MulResult = Builder.CreateNUWMul(LHS, RHS);
      HigherBits = Builder.CreateLShr(MulResult, 64);
    }
    HigherBits = Builder.CreateIntCast(HigherBits, ResType, IsSigned);

    return HigherBits;
  }

  if (BuiltinID == AArch64::BI__writex18byte ||
      BuiltinID == AArch64::BI__writex18word ||
      BuiltinID == AArch64::BI__writex18dword ||
      BuiltinID == AArch64::BI__writex18qword) {
    // Process the args first
    Value *OffsetArg = EmitScalarExpr(E->getArg(0));
    Value *DataArg = EmitScalarExpr(E->getArg(1));

    // Read x18 as i8*
    llvm::Value *X18 = readX18AsPtr(*this);

    // Store val at x18 + offset
    Value *Offset = Builder.CreateZExt(OffsetArg, Int64Ty);
    Value *Ptr = Builder.CreateGEP(Int8Ty, X18, Offset);
    StoreInst *Store =
        Builder.CreateAlignedStore(DataArg, Ptr, CharUnits::One());
    return Store;
  }

  if (BuiltinID == AArch64::BI__readx18byte ||
      BuiltinID == AArch64::BI__readx18word ||
      BuiltinID == AArch64::BI__readx18dword ||
      BuiltinID == AArch64::BI__readx18qword) {
    // Process the args first
    Value *OffsetArg = EmitScalarExpr(E->getArg(0));

    // Read x18 as i8*
    llvm::Value *X18 = readX18AsPtr(*this);

    // Load x18 + offset
    Value *Offset = Builder.CreateZExt(OffsetArg, Int64Ty);
    Value *Ptr = Builder.CreateGEP(Int8Ty, X18, Offset);
    llvm::Type *IntTy = ConvertType(E->getType());
    LoadInst *Load = Builder.CreateAlignedLoad(IntTy, Ptr, CharUnits::One());
    return Load;
  }

  if (BuiltinID == AArch64::BI__addx18byte ||
      BuiltinID == AArch64::BI__addx18word ||
      BuiltinID == AArch64::BI__addx18dword ||
      BuiltinID == AArch64::BI__addx18qword ||
      BuiltinID == AArch64::BI__incx18byte ||
      BuiltinID == AArch64::BI__incx18word ||
      BuiltinID == AArch64::BI__incx18dword ||
      BuiltinID == AArch64::BI__incx18qword) {
    llvm::Type *IntTy;
    bool isIncrement;
    switch (BuiltinID) {
    case AArch64::BI__incx18byte:
      IntTy = Int8Ty;
      isIncrement = true;
      break;
    case AArch64::BI__incx18word:
      IntTy = Int16Ty;
      isIncrement = true;
      break;
    case AArch64::BI__incx18dword:
      IntTy = Int32Ty;
      isIncrement = true;
      break;
    case AArch64::BI__incx18qword:
      IntTy = Int64Ty;
      isIncrement = true;
      break;
    default:
      IntTy = ConvertType(E->getArg(1)->getType());
      isIncrement = false;
      break;
    }
    // Process the args first
    Value *OffsetArg = EmitScalarExpr(E->getArg(0));
    Value *ValToAdd =
        isIncrement ? ConstantInt::get(IntTy, 1) : EmitScalarExpr(E->getArg(1));

    // Read x18 as i8*
    llvm::Value *X18 = readX18AsPtr(*this);

    // Load x18 + offset
    Value *Offset = Builder.CreateZExt(OffsetArg, Int64Ty);
    Value *Ptr = Builder.CreateGEP(Int8Ty, X18, Offset);
    LoadInst *Load = Builder.CreateAlignedLoad(IntTy, Ptr, CharUnits::One());

    // Add values
    Value *AddResult = Builder.CreateAdd(Load, ValToAdd);

    // Store val at x18 + offset
    StoreInst *Store =
        Builder.CreateAlignedStore(AddResult, Ptr, CharUnits::One());
    return Store;
  }

  if (BuiltinID == AArch64::BI_CopyDoubleFromInt64 ||
      BuiltinID == AArch64::BI_CopyFloatFromInt32 ||
      BuiltinID == AArch64::BI_CopyInt32FromFloat ||
      BuiltinID == AArch64::BI_CopyInt64FromDouble) {
    Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *RetTy = ConvertType(E->getType());
    return Builder.CreateBitCast(Arg, RetTy);
  }

  if (BuiltinID == AArch64::BI_CountLeadingOnes ||
      BuiltinID == AArch64::BI_CountLeadingOnes64 ||
      BuiltinID == AArch64::BI_CountLeadingZeros ||
      BuiltinID == AArch64::BI_CountLeadingZeros64) {
    Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = Arg->getType();

    if (BuiltinID == AArch64::BI_CountLeadingOnes ||
        BuiltinID == AArch64::BI_CountLeadingOnes64)
      Arg = Builder.CreateXor(Arg, Constant::getAllOnesValue(ArgType));

    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);
    Value *Result = Builder.CreateCall(F, {Arg, Builder.getInt1(false)});

    if (BuiltinID == AArch64::BI_CountLeadingOnes64 ||
        BuiltinID == AArch64::BI_CountLeadingZeros64)
      Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
    return Result;
  }

  if (BuiltinID == AArch64::BI_CountLeadingSigns ||
      BuiltinID == AArch64::BI_CountLeadingSigns64) {
    Value *Arg = EmitScalarExpr(E->getArg(0));

    Function *F = (BuiltinID == AArch64::BI_CountLeadingSigns)
                      ? CGM.getIntrinsic(Intrinsic::aarch64_cls)
                      : CGM.getIntrinsic(Intrinsic::aarch64_cls64);

    Value *Result = Builder.CreateCall(F, Arg, "cls");
    if (BuiltinID == AArch64::BI_CountLeadingSigns64)
      Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
    return Result;
  }

  if (BuiltinID == AArch64::BI_CountOneBits ||
      BuiltinID == AArch64::BI_CountOneBits64) {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    Value *Result = Builder.CreateCall(F, ArgValue);
    if (BuiltinID == AArch64::BI_CountOneBits64)
      Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
    return Result;
  }

  if (BuiltinID == AArch64::BI__prefetch) {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *RW = llvm::ConstantInt::get(Int32Ty, 0);
    Value *Locality = ConstantInt::get(Int32Ty, 3);
    Value *Data = llvm::ConstantInt::get(Int32Ty, 1);
    Function *F = CGM.getIntrinsic(Intrinsic::prefetch, Address->getType());
    return Builder.CreateCall(F, {Address, RW, Locality, Data});
  }

  if (BuiltinID == AArch64::BI__hlt) {
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_hlt);
    Builder.CreateCall(F, {EmitScalarExpr(E->getArg(0))});

    // Return 0 for convenience, even though MSVC returns some other undefined
    // value.
    return ConstantInt::get(Builder.getInt32Ty(), 0);
  }

  if (BuiltinID == NEON::BI__builtin_neon_vcvth_bf16_f32)
    return Builder.CreateFPTrunc(
        Builder.CreateBitCast(EmitScalarExpr(E->getArg(0)),
                              Builder.getFloatTy()),
        Builder.getBFloatTy());

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<MSVCIntrin> MsvcIntId =
          translateAarch64ToMsvcIntrin(BuiltinID))
    return EmitMSVCBuiltinExpr(*MsvcIntId, E);

  // Some intrinsics are equivalent - if they are use the base intrinsic ID.
  auto It = llvm::find_if(NEONEquivalentIntrinsicMap, [BuiltinID](auto &P) {
    return P.first == BuiltinID;
  });
  if (It != end(NEONEquivalentIntrinsicMap))
    BuiltinID = It->second;

  // Find out if any arguments are required to be integer constant
  // expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  llvm::SmallVector<Value*, 4> Ops;
  Address PtrOp0 = Address::invalid();
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld1_v:
      case NEON::BI__builtin_neon_vld1q_v:
      case NEON::BI__builtin_neon_vld1_dup_v:
      case NEON::BI__builtin_neon_vld1q_dup_v:
      case NEON::BI__builtin_neon_vld1_lane_v:
      case NEON::BI__builtin_neon_vld1q_lane_v:
      case NEON::BI__builtin_neon_vst1_v:
      case NEON::BI__builtin_neon_vst1q_v:
      case NEON::BI__builtin_neon_vst1_lane_v:
      case NEON::BI__builtin_neon_vst1q_lane_v:
      case NEON::BI__builtin_neon_vldap1_lane_s64:
      case NEON::BI__builtin_neon_vldap1q_lane_s64:
      case NEON::BI__builtin_neon_vstl1_lane_s64:
      case NEON::BI__builtin_neon_vstl1q_lane_s64:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        PtrOp0 = EmitPointerWithAlignment(E->getArg(0));
        Ops.push_back(PtrOp0.emitRawPointer(*this));
        continue;
      }
    }
    Ops.push_back(EmitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  auto SISDMap = ArrayRef(AArch64SISDIntrinsicMap);
  const ARMVectorIntrinsicInfo *Builtin = findARMVectorIntrinsicInMap(
      SISDMap, BuiltinID, AArch64SISDIntrinsicsProvenSorted);

  if (Builtin) {
    Ops.push_back(EmitScalarExpr(E->getArg(E->getNumArgs() - 1)));
    Value *Result = EmitCommonNeonSISDBuiltinExpr(*this, *Builtin, Ops, E);
    assert(Result && "SISD intrinsic should have been handled");
    return Result;
  }

  const Expr *Arg = E->getArg(E->getNumArgs()-1);
  NeonTypeFlags Type(0);
  if (std::optional<llvm::APSInt> Result =
          Arg->getIntegerConstantExpr(getContext()))
    // Determine the type of this overloaded NEON intrinsic.
    Type = NeonTypeFlags(Result->getZExtValue());

  bool usgn = Type.isUnsigned();
  bool quad = Type.isQuad();

  // Handle non-overloaded intrinsics first.
  switch (BuiltinID) {
  default: break;
  case NEON::BI__builtin_neon_vabsh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::fabs, HalfTy), Ops, "vabs");
  case NEON::BI__builtin_neon_vaddq_p128: {
    llvm::Type *Ty = GetNeonType(this, NeonTypeFlags::Poly128);
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[0] =  Builder.CreateXor(Ops[0], Ops[1]);
    llvm::Type *Int128Ty = llvm::Type::getIntNTy(getLLVMContext(), 128);
    return Builder.CreateBitCast(Ops[0], Int128Ty);
  }
  case NEON::BI__builtin_neon_vldrq_p128: {
    llvm::Type *Int128Ty = llvm::Type::getIntNTy(getLLVMContext(), 128);
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    return Builder.CreateAlignedLoad(Int128Ty, Ptr,
                                     CharUnits::fromQuantity(16));
  }
  case NEON::BI__builtin_neon_vstrq_p128: {
    Value *Ptr = Ops[0];
    return Builder.CreateDefaultAlignedStore(EmitScalarExpr(E->getArg(1)), Ptr);
  }
  case NEON::BI__builtin_neon_vcvts_f32_u32:
  case NEON::BI__builtin_neon_vcvtd_f64_u64:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcvts_f32_s32:
  case NEON::BI__builtin_neon_vcvtd_f64_s64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    bool Is64 = Ops[0]->getType()->getPrimitiveSizeInBits() == 64;
    llvm::Type *InTy = Is64 ? Int64Ty : Int32Ty;
    llvm::Type *FTy = Is64 ? DoubleTy : FloatTy;
    Ops[0] = Builder.CreateBitCast(Ops[0], InTy);
    if (usgn)
      return Builder.CreateUIToFP(Ops[0], FTy);
    return Builder.CreateSIToFP(Ops[0], FTy);
  }
  case NEON::BI__builtin_neon_vcvth_f16_u16:
  case NEON::BI__builtin_neon_vcvth_f16_u32:
  case NEON::BI__builtin_neon_vcvth_f16_u64:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcvth_f16_s16:
  case NEON::BI__builtin_neon_vcvth_f16_s32:
  case NEON::BI__builtin_neon_vcvth_f16_s64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    llvm::Type *FTy = HalfTy;
    llvm::Type *InTy;
    if (Ops[0]->getType()->getPrimitiveSizeInBits() == 64)
      InTy = Int64Ty;
    else if (Ops[0]->getType()->getPrimitiveSizeInBits() == 32)
      InTy = Int32Ty;
    else
      InTy = Int16Ty;
    Ops[0] = Builder.CreateBitCast(Ops[0], InTy);
    if (usgn)
      return Builder.CreateUIToFP(Ops[0], FTy);
    return Builder.CreateSIToFP(Ops[0], FTy);
  }
  case NEON::BI__builtin_neon_vcvtah_u16_f16:
  case NEON::BI__builtin_neon_vcvtmh_u16_f16:
  case NEON::BI__builtin_neon_vcvtnh_u16_f16:
  case NEON::BI__builtin_neon_vcvtph_u16_f16:
  case NEON::BI__builtin_neon_vcvth_u16_f16:
  case NEON::BI__builtin_neon_vcvtah_s16_f16:
  case NEON::BI__builtin_neon_vcvtmh_s16_f16:
  case NEON::BI__builtin_neon_vcvtnh_s16_f16:
  case NEON::BI__builtin_neon_vcvtph_s16_f16:
  case NEON::BI__builtin_neon_vcvth_s16_f16: {
    unsigned Int;
    llvm::Type *InTy = Int16Ty;
    llvm::Type* FTy  = HalfTy;
    llvm::Type *Tys[2] = {InTy, FTy};
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vcvtah_u16_f16:
      Int = Intrinsic::aarch64_neon_fcvtau; break;
    case NEON::BI__builtin_neon_vcvtmh_u16_f16:
      Int = Intrinsic::aarch64_neon_fcvtmu; break;
    case NEON::BI__builtin_neon_vcvtnh_u16_f16:
      Int = Intrinsic::aarch64_neon_fcvtnu; break;
    case NEON::BI__builtin_neon_vcvtph_u16_f16:
      Int = Intrinsic::aarch64_neon_fcvtpu; break;
    case NEON::BI__builtin_neon_vcvth_u16_f16:
      Int = Intrinsic::aarch64_neon_fcvtzu; break;
    case NEON::BI__builtin_neon_vcvtah_s16_f16:
      Int = Intrinsic::aarch64_neon_fcvtas; break;
    case NEON::BI__builtin_neon_vcvtmh_s16_f16:
      Int = Intrinsic::aarch64_neon_fcvtms; break;
    case NEON::BI__builtin_neon_vcvtnh_s16_f16:
      Int = Intrinsic::aarch64_neon_fcvtns; break;
    case NEON::BI__builtin_neon_vcvtph_s16_f16:
      Int = Intrinsic::aarch64_neon_fcvtps; break;
    case NEON::BI__builtin_neon_vcvth_s16_f16:
      Int = Intrinsic::aarch64_neon_fcvtzs; break;
    }
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "fcvt");
  }
  case NEON::BI__builtin_neon_vcaleh_f16:
  case NEON::BI__builtin_neon_vcalth_f16:
  case NEON::BI__builtin_neon_vcageh_f16:
  case NEON::BI__builtin_neon_vcagth_f16: {
    unsigned Int;
    llvm::Type* InTy = Int32Ty;
    llvm::Type* FTy  = HalfTy;
    llvm::Type *Tys[2] = {InTy, FTy};
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vcageh_f16:
      Int = Intrinsic::aarch64_neon_facge; break;
    case NEON::BI__builtin_neon_vcagth_f16:
      Int = Intrinsic::aarch64_neon_facgt; break;
    case NEON::BI__builtin_neon_vcaleh_f16:
      Int = Intrinsic::aarch64_neon_facge; std::swap(Ops[0], Ops[1]); break;
    case NEON::BI__builtin_neon_vcalth_f16:
      Int = Intrinsic::aarch64_neon_facgt; std::swap(Ops[0], Ops[1]); break;
    }
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "facg");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vcvth_n_s16_f16:
  case NEON::BI__builtin_neon_vcvth_n_u16_f16: {
    unsigned Int;
    llvm::Type* InTy = Int32Ty;
    llvm::Type* FTy  = HalfTy;
    llvm::Type *Tys[2] = {InTy, FTy};
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vcvth_n_s16_f16:
      Int = Intrinsic::aarch64_neon_vcvtfp2fxs; break;
    case NEON::BI__builtin_neon_vcvth_n_u16_f16:
      Int = Intrinsic::aarch64_neon_vcvtfp2fxu; break;
    }
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "fcvth_n");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vcvth_n_f16_s16:
  case NEON::BI__builtin_neon_vcvth_n_f16_u16: {
    unsigned Int;
    llvm::Type* FTy  = HalfTy;
    llvm::Type* InTy = Int32Ty;
    llvm::Type *Tys[2] = {FTy, InTy};
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vcvth_n_f16_s16:
      Int = Intrinsic::aarch64_neon_vcvtfxs2fp;
      Ops[0] = Builder.CreateSExt(Ops[0], InTy, "sext");
      break;
    case NEON::BI__builtin_neon_vcvth_n_f16_u16:
      Int = Intrinsic::aarch64_neon_vcvtfxu2fp;
      Ops[0] = Builder.CreateZExt(Ops[0], InTy);
      break;
    }
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "fcvth_n");
  }
  case NEON::BI__builtin_neon_vpaddd_s64: {
    auto *Ty = llvm::FixedVectorType::get(Int64Ty, 2);
    Value *Vec = EmitScalarExpr(E->getArg(0));
    // The vector is v2f64, so make sure it's bitcast to that.
    Vec = Builder.CreateBitCast(Vec, Ty, "v2i64");
    llvm::Value *Idx0 = llvm::ConstantInt::get(SizeTy, 0);
    llvm::Value *Idx1 = llvm::ConstantInt::get(SizeTy, 1);
    Value *Op0 = Builder.CreateExtractElement(Vec, Idx0, "lane0");
    Value *Op1 = Builder.CreateExtractElement(Vec, Idx1, "lane1");
    // Pairwise addition of a v2f64 into a scalar f64.
    return Builder.CreateAdd(Op0, Op1, "vpaddd");
  }
  case NEON::BI__builtin_neon_vpaddd_f64: {
    auto *Ty = llvm::FixedVectorType::get(DoubleTy, 2);
    Value *Vec = EmitScalarExpr(E->getArg(0));
    // The vector is v2f64, so make sure it's bitcast to that.
    Vec = Builder.CreateBitCast(Vec, Ty, "v2f64");
    llvm::Value *Idx0 = llvm::ConstantInt::get(SizeTy, 0);
    llvm::Value *Idx1 = llvm::ConstantInt::get(SizeTy, 1);
    Value *Op0 = Builder.CreateExtractElement(Vec, Idx0, "lane0");
    Value *Op1 = Builder.CreateExtractElement(Vec, Idx1, "lane1");
    // Pairwise addition of a v2f64 into a scalar f64.
    return Builder.CreateFAdd(Op0, Op1, "vpaddd");
  }
  case NEON::BI__builtin_neon_vpadds_f32: {
    auto *Ty = llvm::FixedVectorType::get(FloatTy, 2);
    Value *Vec = EmitScalarExpr(E->getArg(0));
    // The vector is v2f32, so make sure it's bitcast to that.
    Vec = Builder.CreateBitCast(Vec, Ty, "v2f32");
    llvm::Value *Idx0 = llvm::ConstantInt::get(SizeTy, 0);
    llvm::Value *Idx1 = llvm::ConstantInt::get(SizeTy, 1);
    Value *Op0 = Builder.CreateExtractElement(Vec, Idx0, "lane0");
    Value *Op1 = Builder.CreateExtractElement(Vec, Idx1, "lane1");
    // Pairwise addition of a v2f32 into a scalar f32.
    return Builder.CreateFAdd(Op0, Op1, "vpaddd");
  }
  case NEON::BI__builtin_neon_vceqzd_s64:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::ICMP_EQ, "vceqz");
  case NEON::BI__builtin_neon_vceqzd_f64:
  case NEON::BI__builtin_neon_vceqzs_f32:
  case NEON::BI__builtin_neon_vceqzh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::FCMP_OEQ, "vceqz");
  case NEON::BI__builtin_neon_vcgezd_s64:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::ICMP_SGE, "vcgez");
  case NEON::BI__builtin_neon_vcgezd_f64:
  case NEON::BI__builtin_neon_vcgezs_f32:
  case NEON::BI__builtin_neon_vcgezh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::FCMP_OGE, "vcgez");
  case NEON::BI__builtin_neon_vclezd_s64:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::ICMP_SLE, "vclez");
  case NEON::BI__builtin_neon_vclezd_f64:
  case NEON::BI__builtin_neon_vclezs_f32:
  case NEON::BI__builtin_neon_vclezh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::FCMP_OLE, "vclez");
  case NEON::BI__builtin_neon_vcgtzd_s64:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::ICMP_SGT, "vcgtz");
  case NEON::BI__builtin_neon_vcgtzd_f64:
  case NEON::BI__builtin_neon_vcgtzs_f32:
  case NEON::BI__builtin_neon_vcgtzh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::FCMP_OGT, "vcgtz");
  case NEON::BI__builtin_neon_vcltzd_s64:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::ICMP_SLT, "vcltz");

  case NEON::BI__builtin_neon_vcltzd_f64:
  case NEON::BI__builtin_neon_vcltzs_f32:
  case NEON::BI__builtin_neon_vcltzh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitAArch64CompareBuiltinExpr(
        Ops[0], ConvertType(E->getCallReturnType(getContext())),
        ICmpInst::FCMP_OLT, "vcltz");

  case NEON::BI__builtin_neon_vceqzd_u64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = Builder.CreateBitCast(Ops[0], Int64Ty);
    Ops[0] =
        Builder.CreateICmpEQ(Ops[0], llvm::Constant::getNullValue(Int64Ty));
    return Builder.CreateSExt(Ops[0], Int64Ty, "vceqzd");
  }
  case NEON::BI__builtin_neon_vceqd_f64:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcged_f64:
  case NEON::BI__builtin_neon_vcgtd_f64: {
    llvm::CmpInst::Predicate P;
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vceqd_f64: P = llvm::FCmpInst::FCMP_OEQ; break;
    case NEON::BI__builtin_neon_vcled_f64: P = llvm::FCmpInst::FCMP_OLE; break;
    case NEON::BI__builtin_neon_vcltd_f64: P = llvm::FCmpInst::FCMP_OLT; break;
    case NEON::BI__builtin_neon_vcged_f64: P = llvm::FCmpInst::FCMP_OGE; break;
    case NEON::BI__builtin_neon_vcgtd_f64: P = llvm::FCmpInst::FCMP_OGT; break;
    }
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], DoubleTy);
    if (P == llvm::FCmpInst::FCMP_OEQ)
      Ops[0] = Builder.CreateFCmp(P, Ops[0], Ops[1]);
    else
      Ops[0] = Builder.CreateFCmpS(P, Ops[0], Ops[1]);
    return Builder.CreateSExt(Ops[0], Int64Ty, "vcmpd");
  }
  case NEON::BI__builtin_neon_vceqs_f32:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcges_f32:
  case NEON::BI__builtin_neon_vcgts_f32: {
    llvm::CmpInst::Predicate P;
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vceqs_f32: P = llvm::FCmpInst::FCMP_OEQ; break;
    case NEON::BI__builtin_neon_vcles_f32: P = llvm::FCmpInst::FCMP_OLE; break;
    case NEON::BI__builtin_neon_vclts_f32: P = llvm::FCmpInst::FCMP_OLT; break;
    case NEON::BI__builtin_neon_vcges_f32: P = llvm::FCmpInst::FCMP_OGE; break;
    case NEON::BI__builtin_neon_vcgts_f32: P = llvm::FCmpInst::FCMP_OGT; break;
    }
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], FloatTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], FloatTy);
    if (P == llvm::FCmpInst::FCMP_OEQ)
      Ops[0] = Builder.CreateFCmp(P, Ops[0], Ops[1]);
    else
      Ops[0] = Builder.CreateFCmpS(P, Ops[0], Ops[1]);
    return Builder.CreateSExt(Ops[0], Int32Ty, "vcmpd");
  }
  case NEON::BI__builtin_neon_vceqh_f16:
  case NEON::BI__builtin_neon_vcleh_f16:
  case NEON::BI__builtin_neon_vclth_f16:
  case NEON::BI__builtin_neon_vcgeh_f16:
  case NEON::BI__builtin_neon_vcgth_f16: {
    llvm::CmpInst::Predicate P;
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vceqh_f16: P = llvm::FCmpInst::FCMP_OEQ; break;
    case NEON::BI__builtin_neon_vcleh_f16: P = llvm::FCmpInst::FCMP_OLE; break;
    case NEON::BI__builtin_neon_vclth_f16: P = llvm::FCmpInst::FCMP_OLT; break;
    case NEON::BI__builtin_neon_vcgeh_f16: P = llvm::FCmpInst::FCMP_OGE; break;
    case NEON::BI__builtin_neon_vcgth_f16: P = llvm::FCmpInst::FCMP_OGT; break;
    }
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], HalfTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], HalfTy);
    if (P == llvm::FCmpInst::FCMP_OEQ)
      Ops[0] = Builder.CreateFCmp(P, Ops[0], Ops[1]);
    else
      Ops[0] = Builder.CreateFCmpS(P, Ops[0], Ops[1]);
    return Builder.CreateSExt(Ops[0], Int16Ty, "vcmpd");
  }
  case NEON::BI__builtin_neon_vceqd_s64:
  case NEON::BI__builtin_neon_vceqd_u64:
  case NEON::BI__builtin_neon_vcgtd_s64:
  case NEON::BI__builtin_neon_vcgtd_u64:
  case NEON::BI__builtin_neon_vcltd_s64:
  case NEON::BI__builtin_neon_vcltd_u64:
  case NEON::BI__builtin_neon_vcged_u64:
  case NEON::BI__builtin_neon_vcged_s64:
  case NEON::BI__builtin_neon_vcled_u64:
  case NEON::BI__builtin_neon_vcled_s64: {
    llvm::CmpInst::Predicate P;
    switch (BuiltinID) {
    default: llvm_unreachable("missing builtin ID in switch!");
    case NEON::BI__builtin_neon_vceqd_s64:
    case NEON::BI__builtin_neon_vceqd_u64:P = llvm::ICmpInst::ICMP_EQ;break;
    case NEON::BI__builtin_neon_vcgtd_s64:P = llvm::ICmpInst::ICMP_SGT;break;
    case NEON::BI__builtin_neon_vcgtd_u64:P = llvm::ICmpInst::ICMP_UGT;break;
    case NEON::BI__builtin_neon_vcltd_s64:P = llvm::ICmpInst::ICMP_SLT;break;
    case NEON::BI__builtin_neon_vcltd_u64:P = llvm::ICmpInst::ICMP_ULT;break;
    case NEON::BI__builtin_neon_vcged_u64:P = llvm::ICmpInst::ICMP_UGE;break;
    case NEON::BI__builtin_neon_vcged_s64:P = llvm::ICmpInst::ICMP_SGE;break;
    case NEON::BI__builtin_neon_vcled_u64:P = llvm::ICmpInst::ICMP_ULE;break;
    case NEON::BI__builtin_neon_vcled_s64:P = llvm::ICmpInst::ICMP_SLE;break;
    }
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], Int64Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Int64Ty);
    Ops[0] = Builder.CreateICmp(P, Ops[0], Ops[1]);
    return Builder.CreateSExt(Ops[0], Int64Ty, "vceqd");
  }
  case NEON::BI__builtin_neon_vtstd_s64:
  case NEON::BI__builtin_neon_vtstd_u64: {
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[0] = Builder.CreateBitCast(Ops[0], Int64Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Int64Ty);
    Ops[0] = Builder.CreateAnd(Ops[0], Ops[1]);
    Ops[0] = Builder.CreateICmp(ICmpInst::ICMP_NE, Ops[0],
                                llvm::Constant::getNullValue(Int64Ty));
    return Builder.CreateSExt(Ops[0], Int64Ty, "vtstd");
  }
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");
  case NEON::BI__builtin_neon_vset_lane_f64:
    // The vector type needs a cast for the v1f64 variant.
    Ops[1] =
        Builder.CreateBitCast(Ops[1], llvm::FixedVectorType::get(DoubleTy, 1));
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");
  case NEON::BI__builtin_neon_vset_lane_mf8:
  case NEON::BI__builtin_neon_vsetq_lane_mf8:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    // The input vector type needs a cast to scalar type.
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::Type::getInt8Ty(getLLVMContext()));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");
  case NEON::BI__builtin_neon_vsetq_lane_f64:
    // The vector type needs a cast for the v2f64 variant.
    Ops[1] =
        Builder.CreateBitCast(Ops[1], llvm::FixedVectorType::get(DoubleTy, 2));
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");

  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vdupb_lane_i8:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int8Ty, 8));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vdupb_laneq_i8:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int8Ty, 16));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vget_lane_mf8:
  case NEON::BI__builtin_neon_vdupb_lane_mf8:
  case NEON::BI__builtin_neon_vgetq_lane_mf8:
  case NEON::BI__builtin_neon_vdupb_laneq_mf8:
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vduph_lane_i16:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int16Ty, 4));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vduph_laneq_i16:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int16Ty, 8));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vdups_lane_i32:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int32Ty, 2));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vdups_lane_f32:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(FloatTy, 2));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vdups_lane");
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vdups_laneq_i32:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int32Ty, 4));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vdupd_lane_i64:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int64Ty, 1));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vdupd_lane_f64:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(DoubleTy, 1));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vdupd_lane");
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vdupd_laneq_i64:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(Int64Ty, 2));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vget_lane_f32:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(FloatTy, 2));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vget_lane_f64:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(DoubleTy, 1));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vdups_laneq_f32:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(FloatTy, 4));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vgetq_lane_f64:
  case NEON::BI__builtin_neon_vdupd_laneq_f64:
    Ops[0] =
        Builder.CreateBitCast(Ops[0], llvm::FixedVectorType::get(DoubleTy, 2));
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  case NEON::BI__builtin_neon_vaddh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateFAdd(Ops[0], Ops[1], "vaddh");
  case NEON::BI__builtin_neon_vsubh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateFSub(Ops[0], Ops[1], "vsubh");
  case NEON::BI__builtin_neon_vmulh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateFMul(Ops[0], Ops[1], "vmulh");
  case NEON::BI__builtin_neon_vdivh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateFDiv(Ops[0], Ops[1], "vdivh");
  case NEON::BI__builtin_neon_vfmah_f16:
    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, HalfTy,
        {EmitScalarExpr(E->getArg(1)), EmitScalarExpr(E->getArg(2)), Ops[0]});
  case NEON::BI__builtin_neon_vfmsh_f16: {
    Value* Neg = Builder.CreateFNeg(EmitScalarExpr(E->getArg(1)), "vsubh");

    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, HalfTy,
        {Neg, EmitScalarExpr(E->getArg(2)), Ops[0]});
  }
  case NEON::BI__builtin_neon_vaddd_s64:
  case NEON::BI__builtin_neon_vaddd_u64:
    return Builder.CreateAdd(Ops[0], EmitScalarExpr(E->getArg(1)), "vaddd");
  case NEON::BI__builtin_neon_vsubd_s64:
  case NEON::BI__builtin_neon_vsubd_u64:
    return Builder.CreateSub(Ops[0], EmitScalarExpr(E->getArg(1)), "vsubd");
  case NEON::BI__builtin_neon_vqdmlalh_s16:
  case NEON::BI__builtin_neon_vqdmlslh_s16: {
    SmallVector<Value *, 2> ProductOps;
    ProductOps.push_back(vectorWrapScalar16(Ops[1]));
    ProductOps.push_back(vectorWrapScalar16(EmitScalarExpr(E->getArg(2))));
    auto *VTy = llvm::FixedVectorType::get(Int32Ty, 4);
    Ops[1] = EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_sqdmull, VTy),
                          ProductOps, "vqdmlXl");
    Constant *CI = ConstantInt::get(SizeTy, 0);
    Ops[1] = Builder.CreateExtractElement(Ops[1], CI, "lane0");

    unsigned AccumInt = BuiltinID == NEON::BI__builtin_neon_vqdmlalh_s16
                                        ? Intrinsic::aarch64_neon_sqadd
                                        : Intrinsic::aarch64_neon_sqsub;
    return EmitNeonCall(CGM.getIntrinsic(AccumInt, Int32Ty), Ops, "vqdmlXl");
  }
  case NEON::BI__builtin_neon_vqshlud_n_s64: {
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[1] = Builder.CreateZExt(Ops[1], Int64Ty);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_sqshlu, Int64Ty),
                        Ops, "vqshlu_n");
  }
  case NEON::BI__builtin_neon_vqshld_n_u64:
  case NEON::BI__builtin_neon_vqshld_n_s64: {
    unsigned Int = BuiltinID == NEON::BI__builtin_neon_vqshld_n_u64
                                   ? Intrinsic::aarch64_neon_uqshl
                                   : Intrinsic::aarch64_neon_sqshl;
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops[1] = Builder.CreateZExt(Ops[1], Int64Ty);
    return EmitNeonCall(CGM.getIntrinsic(Int, Int64Ty), Ops, "vqshl_n");
  }
  case NEON::BI__builtin_neon_vrshrd_n_u64:
  case NEON::BI__builtin_neon_vrshrd_n_s64: {
    unsigned Int = BuiltinID == NEON::BI__builtin_neon_vrshrd_n_u64
                                   ? Intrinsic::aarch64_neon_urshl
                                   : Intrinsic::aarch64_neon_srshl;
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    int SV = cast<ConstantInt>(Ops[1])->getSExtValue();
    Ops[1] = ConstantInt::get(Int64Ty, -SV);
    return EmitNeonCall(CGM.getIntrinsic(Int, Int64Ty), Ops, "vrshr_n");
  }
  case NEON::BI__builtin_neon_vrsrad_n_u64:
  case NEON::BI__builtin_neon_vrsrad_n_s64: {
    unsigned Int = BuiltinID == NEON::BI__builtin_neon_vrsrad_n_u64
                                   ? Intrinsic::aarch64_neon_urshl
                                   : Intrinsic::aarch64_neon_srshl;
    Ops[1] = Builder.CreateBitCast(Ops[1], Int64Ty);
    Ops.push_back(Builder.CreateNeg(EmitScalarExpr(E->getArg(2))));
    Ops[1] = Builder.CreateCall(CGM.getIntrinsic(Int, Int64Ty),
                                {Ops[1], Builder.CreateSExt(Ops[2], Int64Ty)});
    return Builder.CreateAdd(Ops[0], Builder.CreateBitCast(Ops[1], Int64Ty));
  }
  case NEON::BI__builtin_neon_vshld_n_s64:
  case NEON::BI__builtin_neon_vshld_n_u64: {
    llvm::ConstantInt *Amt = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateShl(
        Ops[0], ConstantInt::get(Int64Ty, Amt->getZExtValue()), "shld_n");
  }
  case NEON::BI__builtin_neon_vshrd_n_s64: {
    llvm::ConstantInt *Amt = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    return Builder.CreateAShr(
        Ops[0], ConstantInt::get(Int64Ty, std::min(static_cast<uint64_t>(63),
                                                   Amt->getZExtValue())),
        "shrd_n");
  }
  case NEON::BI__builtin_neon_vshrd_n_u64: {
    llvm::ConstantInt *Amt = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    uint64_t ShiftAmt = Amt->getZExtValue();
    // Right-shifting an unsigned value by its size yields 0.
    if (ShiftAmt == 64)
      return ConstantInt::get(Int64Ty, 0);
    return Builder.CreateLShr(Ops[0], ConstantInt::get(Int64Ty, ShiftAmt),
                              "shrd_n");
  }
  case NEON::BI__builtin_neon_vsrad_n_s64: {
    llvm::ConstantInt *Amt = cast<ConstantInt>(EmitScalarExpr(E->getArg(2)));
    Ops[1] = Builder.CreateAShr(
        Ops[1], ConstantInt::get(Int64Ty, std::min(static_cast<uint64_t>(63),
                                                   Amt->getZExtValue())),
        "shrd_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vsrad_n_u64: {
    llvm::ConstantInt *Amt = cast<ConstantInt>(EmitScalarExpr(E->getArg(2)));
    uint64_t ShiftAmt = Amt->getZExtValue();
    // Right-shifting an unsigned value by its size yields 0.
    // As Op + 0 = Op, return Ops[0] directly.
    if (ShiftAmt == 64)
      return Ops[0];
    Ops[1] = Builder.CreateLShr(Ops[1], ConstantInt::get(Int64Ty, ShiftAmt),
                                "shrd_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vqdmlalh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlalh_laneq_s16:
  case NEON::BI__builtin_neon_vqdmlslh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlslh_laneq_s16: {
    Ops[2] = Builder.CreateExtractElement(Ops[2], EmitScalarExpr(E->getArg(3)),
                                          "lane");
    SmallVector<Value *, 2> ProductOps;
    ProductOps.push_back(vectorWrapScalar16(Ops[1]));
    ProductOps.push_back(vectorWrapScalar16(Ops[2]));
    auto *VTy = llvm::FixedVectorType::get(Int32Ty, 4);
    Ops[1] = EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_sqdmull, VTy),
                          ProductOps, "vqdmlXl");
    Constant *CI = ConstantInt::get(SizeTy, 0);
    Ops[1] = Builder.CreateExtractElement(Ops[1], CI, "lane0");
    Ops.pop_back();

    unsigned AccInt = (BuiltinID == NEON::BI__builtin_neon_vqdmlalh_lane_s16 ||
                       BuiltinID == NEON::BI__builtin_neon_vqdmlalh_laneq_s16)
                          ? Intrinsic::aarch64_neon_sqadd
                          : Intrinsic::aarch64_neon_sqsub;
    return EmitNeonCall(CGM.getIntrinsic(AccInt, Int32Ty), Ops, "vqdmlXl");
  }
  case NEON::BI__builtin_neon_vqdmlals_s32:
  case NEON::BI__builtin_neon_vqdmlsls_s32: {
    SmallVector<Value *, 2> ProductOps;
    ProductOps.push_back(Ops[1]);
    ProductOps.push_back(EmitScalarExpr(E->getArg(2)));
    Ops[1] =
        EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_sqdmulls_scalar),
                     ProductOps, "vqdmlXl");

    unsigned AccumInt = BuiltinID == NEON::BI__builtin_neon_vqdmlals_s32
                                        ? Intrinsic::aarch64_neon_sqadd
                                        : Intrinsic::aarch64_neon_sqsub;
    return EmitNeonCall(CGM.getIntrinsic(AccumInt, Int64Ty), Ops, "vqdmlXl");
  }
  case NEON::BI__builtin_neon_vqdmlals_lane_s32:
  case NEON::BI__builtin_neon_vqdmlals_laneq_s32:
  case NEON::BI__builtin_neon_vqdmlsls_lane_s32:
  case NEON::BI__builtin_neon_vqdmlsls_laneq_s32: {
    Ops[2] = Builder.CreateExtractElement(Ops[2], EmitScalarExpr(E->getArg(3)),
                                          "lane");
    SmallVector<Value *, 2> ProductOps;
    ProductOps.push_back(Ops[1]);
    ProductOps.push_back(Ops[2]);
    Ops[1] =
        EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_sqdmulls_scalar),
                     ProductOps, "vqdmlXl");
    Ops.pop_back();

    unsigned AccInt = (BuiltinID == NEON::BI__builtin_neon_vqdmlals_lane_s32 ||
                       BuiltinID == NEON::BI__builtin_neon_vqdmlals_laneq_s32)
                          ? Intrinsic::aarch64_neon_sqadd
                          : Intrinsic::aarch64_neon_sqsub;
    return EmitNeonCall(CGM.getIntrinsic(AccInt, Int64Ty), Ops, "vqdmlXl");
  }
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_f16: {
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  }
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_f16: {
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vgetq_lane");
  }
  case NEON::BI__builtin_neon_vcvt_bf16_f32: {
    llvm::Type *V4F32 = FixedVectorType::get(Builder.getFloatTy(), 4);
    llvm::Type *V4BF16 = FixedVectorType::get(Builder.getBFloatTy(), 4);
    return Builder.CreateFPTrunc(Builder.CreateBitCast(Ops[0], V4F32), V4BF16);
  }
  case NEON::BI__builtin_neon_vcvtq_low_bf16_f32: {
    SmallVector<int, 16> ConcatMask(8);
    std::iota(ConcatMask.begin(), ConcatMask.end(), 0);
    llvm::Type *V4F32 = FixedVectorType::get(Builder.getFloatTy(), 4);
    llvm::Type *V4BF16 = FixedVectorType::get(Builder.getBFloatTy(), 4);
    llvm::Value *Trunc =
        Builder.CreateFPTrunc(Builder.CreateBitCast(Ops[0], V4F32), V4BF16);
    return Builder.CreateShuffleVector(
        Trunc, ConstantAggregateZero::get(V4BF16), ConcatMask);
  }
  case NEON::BI__builtin_neon_vcvtq_high_bf16_f32: {
    SmallVector<int, 16> ConcatMask(8);
    std::iota(ConcatMask.begin(), ConcatMask.end(), 0);
    SmallVector<int, 16> LoMask(4);
    std::iota(LoMask.begin(), LoMask.end(), 0);
    llvm::Type *V4F32 = FixedVectorType::get(Builder.getFloatTy(), 4);
    llvm::Type *V4BF16 = FixedVectorType::get(Builder.getBFloatTy(), 4);
    llvm::Type *V8BF16 = FixedVectorType::get(Builder.getBFloatTy(), 8);
    llvm::Value *Inactive = Builder.CreateShuffleVector(
        Builder.CreateBitCast(Ops[0], V8BF16), LoMask);
    llvm::Value *Trunc =
        Builder.CreateFPTrunc(Builder.CreateBitCast(Ops[1], V4F32), V4BF16);
    return Builder.CreateShuffleVector(Inactive, Trunc, ConcatMask);
  }

  case clang::AArch64::BI_InterlockedAdd:
  case clang::AArch64::BI_InterlockedAdd_acq:
  case clang::AArch64::BI_InterlockedAdd_rel:
  case clang::AArch64::BI_InterlockedAdd_nf:
  case clang::AArch64::BI_InterlockedAdd64:
  case clang::AArch64::BI_InterlockedAdd64_acq:
  case clang::AArch64::BI_InterlockedAdd64_rel:
  case clang::AArch64::BI_InterlockedAdd64_nf: {
    Address DestAddr = CheckAtomicAlignment(*this, E);
    Value *Val = EmitScalarExpr(E->getArg(1));
    llvm::AtomicOrdering Ordering;
    switch (BuiltinID) {
    case clang::AArch64::BI_InterlockedAdd:
    case clang::AArch64::BI_InterlockedAdd64:
      Ordering = llvm::AtomicOrdering::SequentiallyConsistent;
      break;
    case clang::AArch64::BI_InterlockedAdd_acq:
    case clang::AArch64::BI_InterlockedAdd64_acq:
      Ordering = llvm::AtomicOrdering::Acquire;
      break;
    case clang::AArch64::BI_InterlockedAdd_rel:
    case clang::AArch64::BI_InterlockedAdd64_rel:
      Ordering = llvm::AtomicOrdering::Release;
      break;
    case clang::AArch64::BI_InterlockedAdd_nf:
    case clang::AArch64::BI_InterlockedAdd64_nf:
      Ordering = llvm::AtomicOrdering::Monotonic;
      break;
    default:
      llvm_unreachable("missing builtin ID in switch!");
    }
    AtomicRMWInst *RMWI =
        Builder.CreateAtomicRMW(AtomicRMWInst::Add, DestAddr, Val, Ordering);
    return Builder.CreateAdd(RMWI, Val);
  }
  }

  llvm::FixedVectorType *VTy = GetNeonType(this, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return nullptr;

  // Not all intrinsics handled by the common case work for AArch64 yet, so only
  // defer to common code if it's been added to our special map.
  Builtin = findARMVectorIntrinsicInMap(AArch64SIMDIntrinsicMap, BuiltinID,
                                        AArch64SIMDIntrinsicsProvenSorted);

  if (Builtin)
    return EmitCommonNeonBuiltinExpr(
        Builtin->BuiltinID, Builtin->LLVMIntrinsic, Builtin->AltLLVMIntrinsic,
        Builtin->NameHint, Builtin->TypeModifier, E, Ops,
        /*never use addresses*/ Address::invalid(), Address::invalid(), Arch);

  if (Value *V = EmitAArch64TblBuiltinExpr(*this, BuiltinID, E, Ops, Arch))
    return V;

  unsigned Int;
  bool ExtractLow = false;
  bool ExtendLaneArg = false;
  switch (BuiltinID) {
  default: return nullptr;
  case NEON::BI__builtin_neon_vbsl_v:
  case NEON::BI__builtin_neon_vbslq_v: {
    llvm::Type *BitTy = llvm::VectorType::getInteger(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], BitTy, "vbsl");
    Ops[1] = Builder.CreateBitCast(Ops[1], BitTy, "vbsl");
    Ops[2] = Builder.CreateBitCast(Ops[2], BitTy, "vbsl");

    Ops[1] = Builder.CreateAnd(Ops[0], Ops[1], "vbsl");
    Ops[2] = Builder.CreateAnd(Builder.CreateNot(Ops[0]), Ops[2], "vbsl");
    Ops[0] = Builder.CreateOr(Ops[1], Ops[2], "vbsl");
    return Builder.CreateBitCast(Ops[0], Ty);
  }
  case NEON::BI__builtin_neon_vfma_lane_v:
  case NEON::BI__builtin_neon_vfmaq_lane_v: { // Only used for FP types
    // The ARM builtins (and instructions) have the addend as the first
    // operand, but the 'fma' intrinsics have it last. Swap it around here.
    Value *Addend = Ops[0];
    Value *Multiplicand = Ops[1];
    Value *LaneSource = Ops[2];
    Ops[0] = Multiplicand;
    Ops[1] = LaneSource;
    Ops[2] = Addend;

    // Now adjust things to handle the lane access.
    auto *SourceTy = BuiltinID == NEON::BI__builtin_neon_vfmaq_lane_v
                         ? llvm::FixedVectorType::get(VTy->getElementType(),
                                                      VTy->getNumElements() / 2)
                         : VTy;
    llvm::Constant *cst = cast<Constant>(Ops[3]);
    Value *SV = llvm::ConstantVector::getSplat(VTy->getElementCount(), cst);
    Ops[1] = Builder.CreateBitCast(Ops[1], SourceTy);
    Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV, "lane");

    Ops.pop_back();
    Int = Builder.getIsFPConstrained() ? Intrinsic::experimental_constrained_fma
                                       : Intrinsic::fma;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "fmla");
  }
  case NEON::BI__builtin_neon_vfma_laneq_v: {
    auto *VTy = cast<llvm::FixedVectorType>(Ty);
    // v1f64 fma should be mapped to Neon scalar f64 fma
    if (VTy && VTy->getElementType() == DoubleTy) {
      Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
      Ops[1] = Builder.CreateBitCast(Ops[1], DoubleTy);
      llvm::FixedVectorType *VTy =
          GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float64, false, true));
      Ops[2] = Builder.CreateBitCast(Ops[2], VTy);
      Ops[2] = Builder.CreateExtractElement(Ops[2], Ops[3], "extract");
      Value *Result;
      Result = emitCallMaybeConstrainedFPBuiltin(
          *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma,
          DoubleTy, {Ops[1], Ops[2], Ops[0]});
      return Builder.CreateBitCast(Result, Ty);
    }
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    auto *STy = llvm::FixedVectorType::get(VTy->getElementType(),
                                           VTy->getNumElements() * 2);
    Ops[2] = Builder.CreateBitCast(Ops[2], STy);
    Value *SV = llvm::ConstantVector::getSplat(VTy->getElementCount(),
                                               cast<ConstantInt>(Ops[3]));
    Ops[2] = Builder.CreateShuffleVector(Ops[2], Ops[2], SV, "lane");

    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, Ty,
        {Ops[2], Ops[1], Ops[0]});
  }
  case NEON::BI__builtin_neon_vfmaq_laneq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[2] = EmitNeonSplat(Ops[2], cast<ConstantInt>(Ops[3]));
    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, Ty,
        {Ops[2], Ops[1], Ops[0]});
  }
  case NEON::BI__builtin_neon_vfmah_lane_f16:
  case NEON::BI__builtin_neon_vfmas_lane_f32:
  case NEON::BI__builtin_neon_vfmah_laneq_f16:
  case NEON::BI__builtin_neon_vfmas_laneq_f32:
  case NEON::BI__builtin_neon_vfmad_lane_f64:
  case NEON::BI__builtin_neon_vfmad_laneq_f64: {
    Ops.push_back(EmitScalarExpr(E->getArg(3)));
    llvm::Type *Ty = ConvertType(E->getCallReturnType(getContext()));
    Ops[2] = Builder.CreateExtractElement(Ops[2], Ops[3], "extract");
    return emitCallMaybeConstrainedFPBuiltin(
        *this, Intrinsic::fma, Intrinsic::experimental_constrained_fma, Ty,
        {Ops[1], Ops[2], Ops[0]});
  }
  case NEON::BI__builtin_neon_vmull_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_umull : Intrinsic::aarch64_neon_smull;
    if (Type.isPoly()) Int = Intrinsic::aarch64_neon_pmull;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmull");
  case NEON::BI__builtin_neon_vmax_v:
  case NEON::BI__builtin_neon_vmaxq_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_umax : Intrinsic::aarch64_neon_smax;
    if (Ty->isFPOrFPVectorTy()) Int = Intrinsic::aarch64_neon_fmax;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmax");
  case NEON::BI__builtin_neon_vmaxh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Int = Intrinsic::aarch64_neon_fmax;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vmax");
  }
  case NEON::BI__builtin_neon_vmin_v:
  case NEON::BI__builtin_neon_vminq_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_umin : Intrinsic::aarch64_neon_smin;
    if (Ty->isFPOrFPVectorTy()) Int = Intrinsic::aarch64_neon_fmin;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmin");
  case NEON::BI__builtin_neon_vminh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Int = Intrinsic::aarch64_neon_fmin;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vmin");
  }
  case NEON::BI__builtin_neon_vabd_v:
  case NEON::BI__builtin_neon_vabdq_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_uabd : Intrinsic::aarch64_neon_sabd;
    if (Ty->isFPOrFPVectorTy()) Int = Intrinsic::aarch64_neon_fabd;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vabd");
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v: {
    unsigned ArgElts = VTy->getNumElements();
    llvm::IntegerType *EltTy = cast<IntegerType>(VTy->getElementType());
    unsigned BitWidth = EltTy->getBitWidth();
    auto *ArgTy = llvm::FixedVectorType::get(
        llvm::IntegerType::get(getLLVMContext(), BitWidth / 2), 2 * ArgElts);
    llvm::Type* Tys[2] = { VTy, ArgTy };
    Int = usgn ? Intrinsic::aarch64_neon_uaddlp : Intrinsic::aarch64_neon_saddlp;
    SmallVector<llvm::Value*, 1> TmpOps;
    TmpOps.push_back(Ops[1]);
    Function *F = CGM.getIntrinsic(Int, Tys);
    llvm::Value *tmp = EmitNeonCall(F, TmpOps, "vpadal");
    llvm::Value *addend = Builder.CreateBitCast(Ops[0], tmp->getType());
    return Builder.CreateAdd(tmp, addend);
  }
  case NEON::BI__builtin_neon_vpmin_v:
  case NEON::BI__builtin_neon_vpminq_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_uminp : Intrinsic::aarch64_neon_sminp;
    if (Ty->isFPOrFPVectorTy()) Int = Intrinsic::aarch64_neon_fminp;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmin");
  case NEON::BI__builtin_neon_vpmax_v:
  case NEON::BI__builtin_neon_vpmaxq_v:
    // FIXME: improve sharing scheme to cope with 3 alternative LLVM intrinsics.
    Int = usgn ? Intrinsic::aarch64_neon_umaxp : Intrinsic::aarch64_neon_smaxp;
    if (Ty->isFPOrFPVectorTy()) Int = Intrinsic::aarch64_neon_fmaxp;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmax");
  case NEON::BI__builtin_neon_vminnm_v:
  case NEON::BI__builtin_neon_vminnmq_v:
    Int = Intrinsic::aarch64_neon_fminnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vminnm");
  case NEON::BI__builtin_neon_vminnmh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Int = Intrinsic::aarch64_neon_fminnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vminnm");
  case NEON::BI__builtin_neon_vmaxnm_v:
  case NEON::BI__builtin_neon_vmaxnmq_v:
    Int = Intrinsic::aarch64_neon_fmaxnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmaxnm");
  case NEON::BI__builtin_neon_vmaxnmh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Int = Intrinsic::aarch64_neon_fmaxnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vmaxnm");
  case NEON::BI__builtin_neon_vrecpss_f32: {
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_frecps, FloatTy),
                        Ops, "vrecps");
  }
  case NEON::BI__builtin_neon_vrecpsd_f64:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_frecps, DoubleTy),
                        Ops, "vrecps");
  case NEON::BI__builtin_neon_vrecpsh_f16:
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_frecps, HalfTy),
                        Ops, "vrecps");
  case NEON::BI__builtin_neon_vqshrun_n_v:
    Int = Intrinsic::aarch64_neon_sqshrun;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrun_n");
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    Int = Intrinsic::aarch64_neon_sqrshrun;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrun_n");
  case NEON::BI__builtin_neon_vqshrn_n_v:
    Int = usgn ? Intrinsic::aarch64_neon_uqshrn : Intrinsic::aarch64_neon_sqshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrn_n");
  case NEON::BI__builtin_neon_vrshrn_n_v:
    Int = Intrinsic::aarch64_neon_rshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshrn_n");
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    Int = usgn ? Intrinsic::aarch64_neon_uqrshrn : Intrinsic::aarch64_neon_sqrshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrn_n");
  case NEON::BI__builtin_neon_vrndah_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_round
              : Intrinsic::round;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrnda");
  }
  case NEON::BI__builtin_neon_vrnda_v:
  case NEON::BI__builtin_neon_vrndaq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_round
              : Intrinsic::round;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnda");
  }
  case NEON::BI__builtin_neon_vrndih_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_nearbyint
              : Intrinsic::nearbyint;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndi");
  }
  case NEON::BI__builtin_neon_vrndmh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_floor
              : Intrinsic::floor;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndm");
  }
  case NEON::BI__builtin_neon_vrndm_v:
  case NEON::BI__builtin_neon_vrndmq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_floor
              : Intrinsic::floor;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndm");
  }
  case NEON::BI__builtin_neon_vrndnh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_roundeven
              : Intrinsic::roundeven;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndn");
  }
  case NEON::BI__builtin_neon_vrndn_v:
  case NEON::BI__builtin_neon_vrndnq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_roundeven
              : Intrinsic::roundeven;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndn");
  }
  case NEON::BI__builtin_neon_vrndns_f32: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_roundeven
              : Intrinsic::roundeven;
    return EmitNeonCall(CGM.getIntrinsic(Int, FloatTy), Ops, "vrndn");
  }
  case NEON::BI__builtin_neon_vrndph_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_ceil
              : Intrinsic::ceil;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndp");
  }
  case NEON::BI__builtin_neon_vrndp_v:
  case NEON::BI__builtin_neon_vrndpq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_ceil
              : Intrinsic::ceil;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndp");
  }
  case NEON::BI__builtin_neon_vrndxh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_rint
              : Intrinsic::rint;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndx");
  }
  case NEON::BI__builtin_neon_vrndx_v:
  case NEON::BI__builtin_neon_vrndxq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_rint
              : Intrinsic::rint;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndx");
  }
  case NEON::BI__builtin_neon_vrndh_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_trunc
              : Intrinsic::trunc;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vrndz");
  }
  case NEON::BI__builtin_neon_vrnd32x_f32:
  case NEON::BI__builtin_neon_vrnd32xq_f32:
  case NEON::BI__builtin_neon_vrnd32x_f64:
  case NEON::BI__builtin_neon_vrnd32xq_f64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Intrinsic::aarch64_neon_frint32x;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnd32x");
  }
  case NEON::BI__builtin_neon_vrnd32z_f32:
  case NEON::BI__builtin_neon_vrnd32zq_f32:
  case NEON::BI__builtin_neon_vrnd32z_f64:
  case NEON::BI__builtin_neon_vrnd32zq_f64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Intrinsic::aarch64_neon_frint32z;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnd32z");
  }
  case NEON::BI__builtin_neon_vrnd64x_f32:
  case NEON::BI__builtin_neon_vrnd64xq_f32:
  case NEON::BI__builtin_neon_vrnd64x_f64:
  case NEON::BI__builtin_neon_vrnd64xq_f64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Intrinsic::aarch64_neon_frint64x;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnd64x");
  }
  case NEON::BI__builtin_neon_vrnd64z_f32:
  case NEON::BI__builtin_neon_vrnd64zq_f32:
  case NEON::BI__builtin_neon_vrnd64z_f64:
  case NEON::BI__builtin_neon_vrnd64zq_f64: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Intrinsic::aarch64_neon_frint64z;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnd64z");
  }
  case NEON::BI__builtin_neon_vrnd_v:
  case NEON::BI__builtin_neon_vrndq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_trunc
              : Intrinsic::trunc;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndz");
  }
  case NEON::BI__builtin_neon_vcvt_f64_v:
  case NEON::BI__builtin_neon_vcvtq_f64_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float64, false, quad));
    return usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  case NEON::BI__builtin_neon_vcvt_f64_f32: {
    assert(Type.getEltType() == NeonTypeFlags::Float64 && quad &&
           "unexpected vcvt_f64_f32 builtin");
    NeonTypeFlags SrcFlag = NeonTypeFlags(NeonTypeFlags::Float32, false, false);
    Ops[0] = Builder.CreateBitCast(Ops[0], GetNeonType(this, SrcFlag));

    return Builder.CreateFPExt(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvt_f32_f64: {
    assert(Type.getEltType() == NeonTypeFlags::Float32 &&
           "unexpected vcvt_f32_f64 builtin");
    NeonTypeFlags SrcFlag = NeonTypeFlags(NeonTypeFlags::Float64, false, true);
    Ops[0] = Builder.CreateBitCast(Ops[0], GetNeonType(this, SrcFlag));

    return Builder.CreateFPTrunc(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvt_s32_v:
  case NEON::BI__builtin_neon_vcvt_u32_v:
  case NEON::BI__builtin_neon_vcvt_s64_v:
  case NEON::BI__builtin_neon_vcvt_u64_v:
  case NEON::BI__builtin_neon_vcvt_s16_f16:
  case NEON::BI__builtin_neon_vcvt_u16_f16:
  case NEON::BI__builtin_neon_vcvtq_s32_v:
  case NEON::BI__builtin_neon_vcvtq_u32_v:
  case NEON::BI__builtin_neon_vcvtq_s64_v:
  case NEON::BI__builtin_neon_vcvtq_u64_v:
  case NEON::BI__builtin_neon_vcvtq_s16_f16:
  case NEON::BI__builtin_neon_vcvtq_u16_f16: {
    Int =
        usgn ? Intrinsic::aarch64_neon_fcvtzu : Intrinsic::aarch64_neon_fcvtzs;
    llvm::Type *Tys[2] = {Ty, GetFloatNeonType(this, Type)};
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvtz");
  }
  case NEON::BI__builtin_neon_vcvta_s16_f16:
  case NEON::BI__builtin_neon_vcvta_u16_f16:
  case NEON::BI__builtin_neon_vcvta_s32_v:
  case NEON::BI__builtin_neon_vcvtaq_s16_f16:
  case NEON::BI__builtin_neon_vcvtaq_s32_v:
  case NEON::BI__builtin_neon_vcvta_u32_v:
  case NEON::BI__builtin_neon_vcvtaq_u16_f16:
  case NEON::BI__builtin_neon_vcvtaq_u32_v:
  case NEON::BI__builtin_neon_vcvta_s64_v:
  case NEON::BI__builtin_neon_vcvtaq_s64_v:
  case NEON::BI__builtin_neon_vcvta_u64_v:
  case NEON::BI__builtin_neon_vcvtaq_u64_v: {
    Int = usgn ? Intrinsic::aarch64_neon_fcvtau : Intrinsic::aarch64_neon_fcvtas;
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvta");
  }
  case NEON::BI__builtin_neon_vcvtm_s16_f16:
  case NEON::BI__builtin_neon_vcvtm_s32_v:
  case NEON::BI__builtin_neon_vcvtmq_s16_f16:
  case NEON::BI__builtin_neon_vcvtmq_s32_v:
  case NEON::BI__builtin_neon_vcvtm_u16_f16:
  case NEON::BI__builtin_neon_vcvtm_u32_v:
  case NEON::BI__builtin_neon_vcvtmq_u16_f16:
  case NEON::BI__builtin_neon_vcvtmq_u32_v:
  case NEON::BI__builtin_neon_vcvtm_s64_v:
  case NEON::BI__builtin_neon_vcvtmq_s64_v:
  case NEON::BI__builtin_neon_vcvtm_u64_v:
  case NEON::BI__builtin_neon_vcvtmq_u64_v: {
    Int = usgn ? Intrinsic::aarch64_neon_fcvtmu : Intrinsic::aarch64_neon_fcvtms;
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvtm");
  }
  case NEON::BI__builtin_neon_vcvtn_s16_f16:
  case NEON::BI__builtin_neon_vcvtn_s32_v:
  case NEON::BI__builtin_neon_vcvtnq_s16_f16:
  case NEON::BI__builtin_neon_vcvtnq_s32_v:
  case NEON::BI__builtin_neon_vcvtn_u16_f16:
  case NEON::BI__builtin_neon_vcvtn_u32_v:
  case NEON::BI__builtin_neon_vcvtnq_u16_f16:
  case NEON::BI__builtin_neon_vcvtnq_u32_v:
  case NEON::BI__builtin_neon_vcvtn_s64_v:
  case NEON::BI__builtin_neon_vcvtnq_s64_v:
  case NEON::BI__builtin_neon_vcvtn_u64_v:
  case NEON::BI__builtin_neon_vcvtnq_u64_v: {
    Int = usgn ? Intrinsic::aarch64_neon_fcvtnu : Intrinsic::aarch64_neon_fcvtns;
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvtn");
  }
  case NEON::BI__builtin_neon_vcvtp_s16_f16:
  case NEON::BI__builtin_neon_vcvtp_s32_v:
  case NEON::BI__builtin_neon_vcvtpq_s16_f16:
  case NEON::BI__builtin_neon_vcvtpq_s32_v:
  case NEON::BI__builtin_neon_vcvtp_u16_f16:
  case NEON::BI__builtin_neon_vcvtp_u32_v:
  case NEON::BI__builtin_neon_vcvtpq_u16_f16:
  case NEON::BI__builtin_neon_vcvtpq_u32_v:
  case NEON::BI__builtin_neon_vcvtp_s64_v:
  case NEON::BI__builtin_neon_vcvtpq_s64_v:
  case NEON::BI__builtin_neon_vcvtp_u64_v:
  case NEON::BI__builtin_neon_vcvtpq_u64_v: {
    Int = usgn ? Intrinsic::aarch64_neon_fcvtpu : Intrinsic::aarch64_neon_fcvtps;
    llvm::Type *Tys[2] = { Ty, GetFloatNeonType(this, Type) };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvtp");
  }
  case NEON::BI__builtin_neon_vmulx_v:
  case NEON::BI__builtin_neon_vmulxq_v: {
    Int = Intrinsic::aarch64_neon_fmulx;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmulx");
  }
  case NEON::BI__builtin_neon_vmulxh_lane_f16:
  case NEON::BI__builtin_neon_vmulxh_laneq_f16: {
    // vmulx_lane should be mapped to Neon scalar mulx after
    // extracting the scalar element
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2], "extract");
    Ops.pop_back();
    Int = Intrinsic::aarch64_neon_fmulx;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vmulx");
  }
  case NEON::BI__builtin_neon_vmul_lane_v:
  case NEON::BI__builtin_neon_vmul_laneq_v: {
    // v1f64 vmul_lane should be mapped to Neon scalar mul lane
    bool Quad = false;
    if (BuiltinID == NEON::BI__builtin_neon_vmul_laneq_v)
      Quad = true;
    Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
    llvm::FixedVectorType *VTy =
        GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float64, false, Quad));
    Ops[1] = Builder.CreateBitCast(Ops[1], VTy);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2], "extract");
    Value *Result = Builder.CreateFMul(Ops[0], Ops[1]);
    return Builder.CreateBitCast(Result, Ty);
  }
  case NEON::BI__builtin_neon_vnegd_s64:
    return Builder.CreateNeg(EmitScalarExpr(E->getArg(0)), "vnegd");
  case NEON::BI__builtin_neon_vnegh_f16:
    return Builder.CreateFNeg(EmitScalarExpr(E->getArg(0)), "vnegh");
  case NEON::BI__builtin_neon_vpmaxnm_v:
  case NEON::BI__builtin_neon_vpmaxnmq_v: {
    Int = Intrinsic::aarch64_neon_fmaxnmp;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmaxnm");
  }
  case NEON::BI__builtin_neon_vpminnm_v:
  case NEON::BI__builtin_neon_vpminnmq_v: {
    Int = Intrinsic::aarch64_neon_fminnmp;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpminnm");
  }
  case NEON::BI__builtin_neon_vsqrth_f16: {
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_sqrt
              : Intrinsic::sqrt;
    return EmitNeonCall(CGM.getIntrinsic(Int, HalfTy), Ops, "vsqrt");
  }
  case NEON::BI__builtin_neon_vsqrt_v:
  case NEON::BI__builtin_neon_vsqrtq_v: {
    Int = Builder.getIsFPConstrained()
              ? Intrinsic::experimental_constrained_sqrt
              : Intrinsic::sqrt;
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsqrt");
  }
  case NEON::BI__builtin_neon_vrbit_v:
  case NEON::BI__builtin_neon_vrbitq_v: {
    Int = Intrinsic::bitreverse;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrbit");
  }
  case NEON::BI__builtin_neon_vaddv_u8:
    // FIXME: These are handled by the AArch64 scalar code.
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s8: {
    Int = usgn ? Intrinsic::aarch64_neon_uaddv : Intrinsic::aarch64_neon_saddv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vaddv_u16:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s16: {
    Int = usgn ? Intrinsic::aarch64_neon_uaddv : Intrinsic::aarch64_neon_saddv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vaddvq_u8:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s8: {
    Int = usgn ? Intrinsic::aarch64_neon_uaddv : Intrinsic::aarch64_neon_saddv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vaddvq_u16:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s16: {
    Int = usgn ? Intrinsic::aarch64_neon_uaddv : Intrinsic::aarch64_neon_saddv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vmaxv_u8: {
    Int = Intrinsic::aarch64_neon_umaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vmaxv_u16: {
    Int = Intrinsic::aarch64_neon_umaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vmaxvq_u8: {
    Int = Intrinsic::aarch64_neon_umaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vmaxvq_u16: {
    Int = Intrinsic::aarch64_neon_umaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vmaxv_s8: {
    Int = Intrinsic::aarch64_neon_smaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vmaxv_s16: {
    Int = Intrinsic::aarch64_neon_smaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vmaxvq_s8: {
    Int = Intrinsic::aarch64_neon_smaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vmaxvq_s16: {
    Int = Intrinsic::aarch64_neon_smaxv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vmaxv_f16: {
    Int = Intrinsic::aarch64_neon_fmaxv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vmaxvq_f16: {
    Int = Intrinsic::aarch64_neon_fmaxv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vminv_u8: {
    Int = Intrinsic::aarch64_neon_uminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vminv_u16: {
    Int = Intrinsic::aarch64_neon_uminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vminvq_u8: {
    Int = Intrinsic::aarch64_neon_uminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vminvq_u16: {
    Int = Intrinsic::aarch64_neon_uminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vminv_s8: {
    Int = Intrinsic::aarch64_neon_sminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vminv_s16: {
    Int = Intrinsic::aarch64_neon_sminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vminvq_s8: {
    Int = Intrinsic::aarch64_neon_sminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int8Ty);
  }
  case NEON::BI__builtin_neon_vminvq_s16: {
    Int = Intrinsic::aarch64_neon_sminv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vminv_f16: {
    Int = Intrinsic::aarch64_neon_fminv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vminvq_f16: {
    Int = Intrinsic::aarch64_neon_fminv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vmaxnmv_f16: {
    Int = Intrinsic::aarch64_neon_fmaxnmv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxnmv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vmaxnmvq_f16: {
    Int = Intrinsic::aarch64_neon_fmaxnmv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vmaxnmv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vminnmv_f16: {
    Int = Intrinsic::aarch64_neon_fminnmv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminnmv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vminnmvq_f16: {
    Int = Intrinsic::aarch64_neon_fminnmv;
    Ty = HalfTy;
    VTy = llvm::FixedVectorType::get(HalfTy, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vminnmv");
    return Builder.CreateTrunc(Ops[0], HalfTy);
  }
  case NEON::BI__builtin_neon_vmul_n_f64: {
    Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
    Value *RHS = Builder.CreateBitCast(EmitScalarExpr(E->getArg(1)), DoubleTy);
    return Builder.CreateFMul(Ops[0], RHS);
  }
  case NEON::BI__builtin_neon_vaddlv_u8: {
    Int = Intrinsic::aarch64_neon_uaddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vaddlv_u16: {
    Int = Intrinsic::aarch64_neon_uaddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
  }
  case NEON::BI__builtin_neon_vaddlvq_u8: {
    Int = Intrinsic::aarch64_neon_uaddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vaddlvq_u16: {
    Int = Intrinsic::aarch64_neon_uaddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
  }
  case NEON::BI__builtin_neon_vaddlv_s8: {
    Int = Intrinsic::aarch64_neon_saddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vaddlv_s16: {
    Int = Intrinsic::aarch64_neon_saddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 4);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
  }
  case NEON::BI__builtin_neon_vaddlvq_s8: {
    Int = Intrinsic::aarch64_neon_saddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int8Ty, 16);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops[0] = EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
    return Builder.CreateTrunc(Ops[0], Int16Ty);
  }
  case NEON::BI__builtin_neon_vaddlvq_s16: {
    Int = Intrinsic::aarch64_neon_saddlv;
    Ty = Int32Ty;
    VTy = llvm::FixedVectorType::get(Int16Ty, 8);
    llvm::Type *Tys[2] = { Ty, VTy };
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vaddlv");
  }
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v: {
    Int = Intrinsic::aarch64_neon_vsri;
    llvm::Function *Intrin = CGM.getIntrinsic(Int, Ty);
    return EmitNeonCall(Intrin, Ops, "vsri_n");
  }
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v: {
    Int = Intrinsic::aarch64_neon_vsli;
    llvm::Function *Intrin = CGM.getIntrinsic(Int, Ty);
    return EmitNeonCall(Intrin, Ops, "vsli_n");
  }
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = EmitNeonRShiftImm(Ops[1], Ops[2], Ty, usgn, "vsra_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v: {
    Int = usgn ? Intrinsic::aarch64_neon_urshl : Intrinsic::aarch64_neon_srshl;
    SmallVector<llvm::Value*,2> TmpOps;
    TmpOps.push_back(Ops[1]);
    TmpOps.push_back(Ops[2]);
    Function* F = CGM.getIntrinsic(Int, Ty);
    llvm::Value *tmp = EmitNeonCall(F, TmpOps, "vrshr_n", 1, true);
    Ops[0] = Builder.CreateBitCast(Ops[0], VTy);
    return Builder.CreateAdd(Ops[0], tmp);
  }
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v: {
    return Builder.CreateAlignedLoad(VTy, Ops[0], PtrOp0.getAlignment());
  }
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v:
    Ops[1] = Builder.CreateBitCast(Ops[1], VTy);
    return Builder.CreateAlignedStore(Ops[1], Ops[0], PtrOp0.getAlignment());
  case NEON::BI__builtin_neon_vld1_lane_v:
  case NEON::BI__builtin_neon_vld1q_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[0] = Builder.CreateAlignedLoad(VTy->getElementType(), Ops[0],
                                       PtrOp0.getAlignment());
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vld1_lane");
  }
  case NEON::BI__builtin_neon_vldap1_lane_s64:
  case NEON::BI__builtin_neon_vldap1q_lane_s64: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    llvm::LoadInst *LI = Builder.CreateAlignedLoad(
        VTy->getElementType(), Ops[0], PtrOp0.getAlignment());
    LI->setAtomic(llvm::AtomicOrdering::Acquire);
    Ops[0] = LI;
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vldap1_lane");
  }
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v: {
    Value *V = PoisonValue::get(Ty);
    Ops[0] = Builder.CreateAlignedLoad(VTy->getElementType(), Ops[0],
                                       PtrOp0.getAlignment());
    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Ops[0] = Builder.CreateInsertElement(V, Ops[0], CI);
    return EmitNeonSplat(Ops[0], CI);
  }
  case NEON::BI__builtin_neon_vst1_lane_v:
  case NEON::BI__builtin_neon_vst1q_lane_v:
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    return Builder.CreateAlignedStore(Ops[1], Ops[0], PtrOp0.getAlignment());
  case NEON::BI__builtin_neon_vstl1_lane_s64:
  case NEON::BI__builtin_neon_vstl1q_lane_s64: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    llvm::StoreInst *SI =
        Builder.CreateAlignedStore(Ops[1], Ops[0], PtrOp0.getAlignment());
    SI->setAtomic(llvm::AtomicOrdering::Release);
    return SI;
  }
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld2, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld2");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld3, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld3");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld4, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld4");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld2r, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld2");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld3r, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld3");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v: {
    llvm::Type *Tys[2] = {VTy, UnqualPtrTy};
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld4r, Tys);
    Ops[1] = Builder.CreateCall(F, Ops[1], "vld4");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v: {
    llvm::Type *Tys[2] = { VTy, Ops[1]->getType() };
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld2lane, Tys);
    std::rotate(Ops.begin() + 1, Ops.begin() + 2, Ops.end());
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateZExt(Ops[3], Int64Ty);
    Ops[1] = Builder.CreateCall(F, ArrayRef(Ops).slice(1), "vld2_lane");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v: {
    llvm::Type *Tys[2] = { VTy, Ops[1]->getType() };
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld3lane, Tys);
    std::rotate(Ops.begin() + 1, Ops.begin() + 2, Ops.end());
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateBitCast(Ops[3], Ty);
    Ops[4] = Builder.CreateZExt(Ops[4], Int64Ty);
    Ops[1] = Builder.CreateCall(F, ArrayRef(Ops).slice(1), "vld3_lane");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v: {
    llvm::Type *Tys[2] = { VTy, Ops[1]->getType() };
    Function *F = CGM.getIntrinsic(Intrinsic::aarch64_neon_ld4lane, Tys);
    std::rotate(Ops.begin() + 1, Ops.begin() + 2, Ops.end());
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateBitCast(Ops[3], Ty);
    Ops[4] = Builder.CreateBitCast(Ops[4], Ty);
    Ops[5] = Builder.CreateZExt(Ops[5], Int64Ty);
    Ops[1] = Builder.CreateCall(F, ArrayRef(Ops).slice(1), "vld4_lane");
    return Builder.CreateDefaultAlignedStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    llvm::Type *Tys[2] = { VTy, Ops[2]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st2, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    Ops[2] = Builder.CreateZExt(Ops[2], Int64Ty);
    llvm::Type *Tys[2] = { VTy, Ops[3]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st2lane, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    llvm::Type *Tys[2] = { VTy, Ops[3]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st3, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    Ops[3] = Builder.CreateZExt(Ops[3], Int64Ty);
    llvm::Type *Tys[2] = { VTy, Ops[4]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st3lane, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    llvm::Type *Tys[2] = { VTy, Ops[4]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st4, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v: {
    std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());
    Ops[4] = Builder.CreateZExt(Ops[4], Int64Ty);
    llvm::Type *Tys[2] = { VTy, Ops[5]->getType() };
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_st4lane, Tys),
                        Ops, "");
  }
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(i+vi);
        Indices.push_back(i+e+vi);
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vtrn");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
        Indices.push_back(2*i+vi);

      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vuzp");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = nullptr;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<int, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back((i + vi*e) >> 1);
        Indices.push_back(((i + vi*e) >> 1)+e);
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ty, Ops[0], vi);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], Indices, "vzip");
      SV = Builder.CreateDefaultAlignedStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vqtbl1q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbl1, Ty),
                        Ops, "vtbl1");
  }
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbl2, Ty),
                        Ops, "vtbl2");
  }
  case NEON::BI__builtin_neon_vqtbl3q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbl3, Ty),
                        Ops, "vtbl3");
  }
  case NEON::BI__builtin_neon_vqtbl4q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbl4, Ty),
                        Ops, "vtbl4");
  }
  case NEON::BI__builtin_neon_vqtbx1q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbx1, Ty),
                        Ops, "vtbx1");
  }
  case NEON::BI__builtin_neon_vqtbx2q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbx2, Ty),
                        Ops, "vtbx2");
  }
  case NEON::BI__builtin_neon_vqtbx3q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbx3, Ty),
                        Ops, "vtbx3");
  }
  case NEON::BI__builtin_neon_vqtbx4q_v: {
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::aarch64_neon_tbx4, Ty),
                        Ops, "vtbx4");
  }
  case NEON::BI__builtin_neon_vsqadd_v:
  case NEON::BI__builtin_neon_vsqaddq_v: {
    Int = Intrinsic::aarch64_neon_usqadd;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsqadd");
  }
  case NEON::BI__builtin_neon_vuqadd_v:
  case NEON::BI__builtin_neon_vuqaddq_v: {
    Int = Intrinsic::aarch64_neon_suqadd;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vuqadd");
  }

  case NEON::BI__builtin_neon_vluti2_laneq_mf8:
  case NEON::BI__builtin_neon_vluti2_laneq_bf16:
  case NEON::BI__builtin_neon_vluti2_laneq_f16:
  case NEON::BI__builtin_neon_vluti2_laneq_p16:
  case NEON::BI__builtin_neon_vluti2_laneq_p8:
  case NEON::BI__builtin_neon_vluti2_laneq_s16:
  case NEON::BI__builtin_neon_vluti2_laneq_s8:
  case NEON::BI__builtin_neon_vluti2_laneq_u16:
  case NEON::BI__builtin_neon_vluti2_laneq_u8: {
    Int = Intrinsic::aarch64_neon_vluti2_laneq;
    llvm::Type *Tys[2];
    Tys[0] = Ty;
    Tys[1] = GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                             /*isQuad*/ false));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vluti2_laneq");
  }
  case NEON::BI__builtin_neon_vluti2q_laneq_mf8:
  case NEON::BI__builtin_neon_vluti2q_laneq_bf16:
  case NEON::BI__builtin_neon_vluti2q_laneq_f16:
  case NEON::BI__builtin_neon_vluti2q_laneq_p16:
  case NEON::BI__builtin_neon_vluti2q_laneq_p8:
  case NEON::BI__builtin_neon_vluti2q_laneq_s16:
  case NEON::BI__builtin_neon_vluti2q_laneq_s8:
  case NEON::BI__builtin_neon_vluti2q_laneq_u16:
  case NEON::BI__builtin_neon_vluti2q_laneq_u8: {
    Int = Intrinsic::aarch64_neon_vluti2_laneq;
    llvm::Type *Tys[2];
    Tys[0] = Ty;
    Tys[1] = GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                             /*isQuad*/ true));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vluti2_laneq");
  }
  case NEON::BI__builtin_neon_vluti2_lane_mf8:
  case NEON::BI__builtin_neon_vluti2_lane_bf16:
  case NEON::BI__builtin_neon_vluti2_lane_f16:
  case NEON::BI__builtin_neon_vluti2_lane_p16:
  case NEON::BI__builtin_neon_vluti2_lane_p8:
  case NEON::BI__builtin_neon_vluti2_lane_s16:
  case NEON::BI__builtin_neon_vluti2_lane_s8:
  case NEON::BI__builtin_neon_vluti2_lane_u16:
  case NEON::BI__builtin_neon_vluti2_lane_u8: {
    Int = Intrinsic::aarch64_neon_vluti2_lane;
    llvm::Type *Tys[2];
    Tys[0] = Ty;
    Tys[1] = GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                             /*isQuad*/ false));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vluti2_lane");
  }
  case NEON::BI__builtin_neon_vluti2q_lane_mf8:
  case NEON::BI__builtin_neon_vluti2q_lane_bf16:
  case NEON::BI__builtin_neon_vluti2q_lane_f16:
  case NEON::BI__builtin_neon_vluti2q_lane_p16:
  case NEON::BI__builtin_neon_vluti2q_lane_p8:
  case NEON::BI__builtin_neon_vluti2q_lane_s16:
  case NEON::BI__builtin_neon_vluti2q_lane_s8:
  case NEON::BI__builtin_neon_vluti2q_lane_u16:
  case NEON::BI__builtin_neon_vluti2q_lane_u8: {
    Int = Intrinsic::aarch64_neon_vluti2_lane;
    llvm::Type *Tys[2];
    Tys[0] = Ty;
    Tys[1] = GetNeonType(this, NeonTypeFlags(Type.getEltType(), false,
                                             /*isQuad*/ true));
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vluti2_lane");
  }
  case NEON::BI__builtin_neon_vluti4q_lane_mf8:
  case NEON::BI__builtin_neon_vluti4q_lane_p8:
  case NEON::BI__builtin_neon_vluti4q_lane_s8:
  case NEON::BI__builtin_neon_vluti4q_lane_u8: {
    Int = Intrinsic::aarch64_neon_vluti4q_lane;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vluti4q_lane");
  }
  case NEON::BI__builtin_neon_vluti4q_laneq_mf8:
  case NEON::BI__builtin_neon_vluti4q_laneq_p8:
  case NEON::BI__builtin_neon_vluti4q_laneq_s8:
  case NEON::BI__builtin_neon_vluti4q_laneq_u8: {
    Int = Intrinsic::aarch64_neon_vluti4q_laneq;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vluti4q_laneq");
  }
  case NEON::BI__builtin_neon_vluti4q_lane_bf16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_f16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_p16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_s16_x2:
  case NEON::BI__builtin_neon_vluti4q_lane_u16_x2: {
    Int = Intrinsic::aarch64_neon_vluti4q_lane_x2;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vluti4q_lane_x2");
  }
  case NEON::BI__builtin_neon_vluti4q_laneq_bf16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_f16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_p16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_s16_x2:
  case NEON::BI__builtin_neon_vluti4q_laneq_u16_x2: {
    Int = Intrinsic::aarch64_neon_vluti4q_laneq_x2;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vluti4q_laneq_x2");
  }
  case NEON::BI__builtin_neon_vcvt1_low_bf16_mf8_fpm:
    ExtractLow = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vcvt1_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_high_bf16_mf8_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_cvtl1,
                              llvm::FixedVectorType::get(BFloatTy, 8),
                              Ops[0]->getType(), ExtractLow, Ops, E, "vbfcvt1");
  case NEON::BI__builtin_neon_vcvt2_low_bf16_mf8_fpm:
    ExtractLow = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vcvt2_bf16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_high_bf16_mf8_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_cvtl2,
                              llvm::FixedVectorType::get(BFloatTy, 8),
                              Ops[0]->getType(), ExtractLow, Ops, E, "vbfcvt2");
  case NEON::BI__builtin_neon_vcvt1_low_f16_mf8_fpm:
    ExtractLow = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vcvt1_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt1_high_f16_mf8_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_cvtl1,
                              llvm::FixedVectorType::get(HalfTy, 8),
                              Ops[0]->getType(), ExtractLow, Ops, E, "vbfcvt1");
  case NEON::BI__builtin_neon_vcvt2_low_f16_mf8_fpm:
    ExtractLow = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vcvt2_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vcvt2_high_f16_mf8_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_cvtl2,
                              llvm::FixedVectorType::get(HalfTy, 8),
                              Ops[0]->getType(), ExtractLow, Ops, E, "vbfcvt2");
  case NEON::BI__builtin_neon_vcvt_mf8_f32_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_fcvtn,
                              llvm::FixedVectorType::get(Int8Ty, 8),
                              Ops[0]->getType(), false, Ops, E, "vfcvtn");
  case NEON::BI__builtin_neon_vcvt_mf8_f16_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_fcvtn,
                              llvm::FixedVectorType::get(Int8Ty, 8),
                              llvm::FixedVectorType::get(HalfTy, 4), false, Ops,
                              E, "vfcvtn");
  case NEON::BI__builtin_neon_vcvtq_mf8_f16_fpm:
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_fcvtn,
                              llvm::FixedVectorType::get(Int8Ty, 16),
                              llvm::FixedVectorType::get(HalfTy, 8), false, Ops,
                              E, "vfcvtn");
  case NEON::BI__builtin_neon_vcvt_high_mf8_f32_fpm: {
    llvm::Type *Ty = llvm::FixedVectorType::get(Int8Ty, 16);
    Ops[0] = Builder.CreateInsertVector(Ty, PoisonValue::get(Ty), Ops[0],
                                        uint64_t(0));
    return EmitFP8NeonCvtCall(Intrinsic::aarch64_neon_fp8_fcvtn2, Ty,
                              Ops[1]->getType(), false, Ops, E, "vfcvtn2");
  }

  case NEON::BI__builtin_neon_vdot_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_f16_mf8_fpm:
    return EmitFP8NeonFDOTCall(Intrinsic::aarch64_neon_fp8_fdot2, false, HalfTy,
                               Ops, E, "fdot2");
  case NEON::BI__builtin_neon_vdot_lane_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_lane_f16_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vdot_laneq_f16_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_laneq_f16_mf8_fpm:
    return EmitFP8NeonFDOTCall(Intrinsic::aarch64_neon_fp8_fdot2_lane,
                               ExtendLaneArg, HalfTy, Ops, E, "fdot2_lane");
  case NEON::BI__builtin_neon_vdot_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_f32_mf8_fpm:
    return EmitFP8NeonFDOTCall(Intrinsic::aarch64_neon_fp8_fdot4, false,
                               FloatTy, Ops, E, "fdot4");
  case NEON::BI__builtin_neon_vdot_lane_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_lane_f32_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vdot_laneq_f32_mf8_fpm:
  case NEON::BI__builtin_neon_vdotq_laneq_f32_mf8_fpm:
    return EmitFP8NeonFDOTCall(Intrinsic::aarch64_neon_fp8_fdot4_lane,
                               ExtendLaneArg, FloatTy, Ops, E, "fdot4_lane");

  case NEON::BI__builtin_neon_vmlalbq_f16_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlalb,
                           {llvm::FixedVectorType::get(HalfTy, 8)}, Ops, E,
                           "vmlal");
  case NEON::BI__builtin_neon_vmlaltq_f16_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlalt,
                           {llvm::FixedVectorType::get(HalfTy, 8)}, Ops, E,
                           "vmlal");
  case NEON::BI__builtin_neon_vmlallbbq_f32_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlallbb,
                           {llvm::FixedVectorType::get(FloatTy, 4)}, Ops, E,
                           "vmlall");
  case NEON::BI__builtin_neon_vmlallbtq_f32_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlallbt,
                           {llvm::FixedVectorType::get(FloatTy, 4)}, Ops, E,
                           "vmlall");
  case NEON::BI__builtin_neon_vmlalltbq_f32_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlalltb,
                           {llvm::FixedVectorType::get(FloatTy, 4)}, Ops, E,
                           "vmlall");
  case NEON::BI__builtin_neon_vmlallttq_f32_mf8_fpm:
    return EmitFP8NeonCall(Intrinsic::aarch64_neon_fp8_fmlalltt,
                           {llvm::FixedVectorType::get(FloatTy, 4)}, Ops, E,
                           "vmlall");
  case NEON::BI__builtin_neon_vmlalbq_lane_f16_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlalbq_laneq_f16_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlalb_lane,
                               ExtendLaneArg, HalfTy, Ops, E, "vmlal_lane");
  case NEON::BI__builtin_neon_vmlaltq_lane_f16_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlaltq_laneq_f16_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlalt_lane,
                               ExtendLaneArg, HalfTy, Ops, E, "vmlal_lane");
  case NEON::BI__builtin_neon_vmlallbbq_lane_f32_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlallbbq_laneq_f32_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlallbb_lane,
                               ExtendLaneArg, FloatTy, Ops, E, "vmlall_lane");
  case NEON::BI__builtin_neon_vmlallbtq_lane_f32_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlallbtq_laneq_f32_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlallbt_lane,
                               ExtendLaneArg, FloatTy, Ops, E, "vmlall_lane");
  case NEON::BI__builtin_neon_vmlalltbq_lane_f32_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlalltbq_laneq_f32_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlalltb_lane,
                               ExtendLaneArg, FloatTy, Ops, E, "vmlall_lane");
  case NEON::BI__builtin_neon_vmlallttq_lane_f32_mf8_fpm:
    ExtendLaneArg = true;
    LLVM_FALLTHROUGH;
  case NEON::BI__builtin_neon_vmlallttq_laneq_f32_mf8_fpm:
    return EmitFP8NeonFMLACall(Intrinsic::aarch64_neon_fp8_fmlalltt_lane,
                               ExtendLaneArg, FloatTy, Ops, E, "vmlall_lane");
  case NEON::BI__builtin_neon_vamin_f16:
  case NEON::BI__builtin_neon_vaminq_f16:
  case NEON::BI__builtin_neon_vamin_f32:
  case NEON::BI__builtin_neon_vaminq_f32:
  case NEON::BI__builtin_neon_vaminq_f64: {
    Int = Intrinsic::aarch64_neon_famin;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "famin");
  }
  case NEON::BI__builtin_neon_vamax_f16:
  case NEON::BI__builtin_neon_vamaxq_f16:
  case NEON::BI__builtin_neon_vamax_f32:
  case NEON::BI__builtin_neon_vamaxq_f32:
  case NEON::BI__builtin_neon_vamaxq_f64: {
    Int = Intrinsic::aarch64_neon_famax;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "famax");
  }
  case NEON::BI__builtin_neon_vscale_f16:
  case NEON::BI__builtin_neon_vscaleq_f16:
  case NEON::BI__builtin_neon_vscale_f32:
  case NEON::BI__builtin_neon_vscaleq_f32:
  case NEON::BI__builtin_neon_vscaleq_f64: {
    Int = Intrinsic::aarch64_neon_fp8_fscale;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "fscale");
  }
  }
}

Value *CodeGenFunction::EmitBPFBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  assert((BuiltinID == BPF::BI__builtin_preserve_field_info ||
          BuiltinID == BPF::BI__builtin_btf_type_id ||
          BuiltinID == BPF::BI__builtin_preserve_type_info ||
          BuiltinID == BPF::BI__builtin_preserve_enum_value) &&
         "unexpected BPF builtin");

  // A sequence number, injected into IR builtin functions, to
  // prevent CSE given the only difference of the function
  // may just be the debuginfo metadata.
  static uint32_t BuiltinSeqNum;

  switch (BuiltinID) {
  default:
    llvm_unreachable("Unexpected BPF builtin");
  case BPF::BI__builtin_preserve_field_info: {
    const Expr *Arg = E->getArg(0);
    bool IsBitField = Arg->IgnoreParens()->getObjectKind() == OK_BitField;

    if (!getDebugInfo()) {
      CGM.Error(E->getExprLoc(),
                "using __builtin_preserve_field_info() without -g");
      return IsBitField ? EmitLValue(Arg).getRawBitFieldPointer(*this)
                        : EmitLValue(Arg).emitRawPointer(*this);
    }

    // Enable underlying preserve_*_access_index() generation.
    bool OldIsInPreservedAIRegion = IsInPreservedAIRegion;
    IsInPreservedAIRegion = true;
    Value *FieldAddr = IsBitField ? EmitLValue(Arg).getRawBitFieldPointer(*this)
                                  : EmitLValue(Arg).emitRawPointer(*this);
    IsInPreservedAIRegion = OldIsInPreservedAIRegion;

    ConstantInt *C = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    Value *InfoKind = ConstantInt::get(Int64Ty, C->getSExtValue());

    // Built the IR for the preserve_field_info intrinsic.
    llvm::Function *FnGetFieldInfo = Intrinsic::getOrInsertDeclaration(
        &CGM.getModule(), Intrinsic::bpf_preserve_field_info,
        {FieldAddr->getType()});
    return Builder.CreateCall(FnGetFieldInfo, {FieldAddr, InfoKind});
  }
  case BPF::BI__builtin_btf_type_id:
  case BPF::BI__builtin_preserve_type_info: {
    if (!getDebugInfo()) {
      CGM.Error(E->getExprLoc(), "using builtin function without -g");
      return nullptr;
    }

    const Expr *Arg0 = E->getArg(0);
    llvm::DIType *DbgInfo = getDebugInfo()->getOrCreateStandaloneType(
        Arg0->getType(), Arg0->getExprLoc());

    ConstantInt *Flag = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    Value *FlagValue = ConstantInt::get(Int64Ty, Flag->getSExtValue());
    Value *SeqNumVal = ConstantInt::get(Int32Ty, BuiltinSeqNum++);

    llvm::Function *FnDecl;
    if (BuiltinID == BPF::BI__builtin_btf_type_id)
      FnDecl = Intrinsic::getOrInsertDeclaration(
          &CGM.getModule(), Intrinsic::bpf_btf_type_id, {});
    else
      FnDecl = Intrinsic::getOrInsertDeclaration(
          &CGM.getModule(), Intrinsic::bpf_preserve_type_info, {});
    CallInst *Fn = Builder.CreateCall(FnDecl, {SeqNumVal, FlagValue});
    Fn->setMetadata(LLVMContext::MD_preserve_access_index, DbgInfo);
    return Fn;
  }
  case BPF::BI__builtin_preserve_enum_value: {
    if (!getDebugInfo()) {
      CGM.Error(E->getExprLoc(), "using builtin function without -g");
      return nullptr;
    }

    const Expr *Arg0 = E->getArg(0);
    llvm::DIType *DbgInfo = getDebugInfo()->getOrCreateStandaloneType(
        Arg0->getType(), Arg0->getExprLoc());

    // Find enumerator
    const auto *UO = cast<UnaryOperator>(Arg0->IgnoreParens());
    const auto *CE = cast<CStyleCastExpr>(UO->getSubExpr());
    const auto *DR = cast<DeclRefExpr>(CE->getSubExpr());
    const auto *Enumerator = cast<EnumConstantDecl>(DR->getDecl());

    auto InitVal = Enumerator->getInitVal();
    std::string InitValStr;
    if (InitVal.isNegative() || InitVal > uint64_t(INT64_MAX))
      InitValStr = std::to_string(InitVal.getSExtValue());
    else
      InitValStr = std::to_string(InitVal.getZExtValue());
    std::string EnumStr = Enumerator->getNameAsString() + ":" + InitValStr;
    Value *EnumStrVal = Builder.CreateGlobalString(EnumStr);

    ConstantInt *Flag = cast<ConstantInt>(EmitScalarExpr(E->getArg(1)));
    Value *FlagValue = ConstantInt::get(Int64Ty, Flag->getSExtValue());
    Value *SeqNumVal = ConstantInt::get(Int32Ty, BuiltinSeqNum++);

    llvm::Function *IntrinsicFn = Intrinsic::getOrInsertDeclaration(
        &CGM.getModule(), Intrinsic::bpf_preserve_enum_value, {});
    CallInst *Fn =
        Builder.CreateCall(IntrinsicFn, {SeqNumVal, EnumStrVal, FlagValue});
    Fn->setMetadata(LLVMContext::MD_preserve_access_index, DbgInfo);
    return Fn;
  }
  }
}

llvm::Value *CodeGenFunction::
BuildVector(ArrayRef<llvm::Value*> Ops) {
  assert((Ops.size() & (Ops.size() - 1)) == 0 &&
         "Not a power-of-two sized vector!");
  bool AllConstants = true;
  for (unsigned i = 0, e = Ops.size(); i != e && AllConstants; ++i)
    AllConstants &= isa<Constant>(Ops[i]);

  // If this is a constant vector, create a ConstantVector.
  if (AllConstants) {
    SmallVector<llvm::Constant*, 16> CstOps;
    for (llvm::Value *Op : Ops)
      CstOps.push_back(cast<Constant>(Op));
    return llvm::ConstantVector::get(CstOps);
  }

  // Otherwise, insertelement the values to build the vector.
  Value *Result = llvm::PoisonValue::get(
      llvm::FixedVectorType::get(Ops[0]->getType(), Ops.size()));

  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    Result = Builder.CreateInsertElement(Result, Ops[i], Builder.getInt64(i));

  return Result;
}

Value *CodeGenFunction::EmitAArch64CpuInit() {
  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, false);
  llvm::FunctionCallee Func =
      CGM.CreateRuntimeFunction(FTy, "__init_cpu_features_resolver");
  cast<llvm::GlobalValue>(Func.getCallee())->setDSOLocal(true);
  cast<llvm::GlobalValue>(Func.getCallee())
      ->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
  return Builder.CreateCall(Func);
}

Value *CodeGenFunction::EmitAArch64CpuSupports(const CallExpr *E) {
  const Expr *ArgExpr = E->getArg(0)->IgnoreParenCasts();
  StringRef ArgStr = cast<StringLiteral>(ArgExpr)->getString();
  llvm::SmallVector<StringRef, 8> Features;
  ArgStr.split(Features, "+");
  for (auto &Feature : Features) {
    Feature = Feature.trim();
    if (!llvm::AArch64::parseFMVExtension(Feature))
      return Builder.getFalse();
    if (Feature != "default")
      Features.push_back(Feature);
  }
  return EmitAArch64CpuSupports(Features);
}

llvm::Value *
CodeGenFunction::EmitAArch64CpuSupports(ArrayRef<StringRef> FeaturesStrs) {
  llvm::APInt FeaturesMask = llvm::AArch64::getCpuSupportsMask(FeaturesStrs);
  Value *Result = Builder.getTrue();
  if (FeaturesMask != 0) {
    // Get features from structure in runtime library
    // struct {
    //   unsigned long long features;
    // } __aarch64_cpu_features;
    llvm::Type *STy = llvm::StructType::get(Int64Ty);
    llvm::Constant *AArch64CPUFeatures =
        CGM.CreateRuntimeVariable(STy, "__aarch64_cpu_features");
    cast<llvm::GlobalValue>(AArch64CPUFeatures)->setDSOLocal(true);
    llvm::Value *CpuFeatures = Builder.CreateGEP(
        STy, AArch64CPUFeatures,
        {ConstantInt::get(Int32Ty, 0), ConstantInt::get(Int32Ty, 0)});
    Value *Features = Builder.CreateAlignedLoad(Int64Ty, CpuFeatures,
                                                CharUnits::fromQuantity(8));
    Value *Mask = Builder.getInt(FeaturesMask.trunc(64));
    Value *Bitset = Builder.CreateAnd(Features, Mask);
    Value *Cmp = Builder.CreateICmpEQ(Bitset, Mask);
    Result = Builder.CreateAnd(Result, Cmp);
  }
  return Result;
}
