//===---------- X86.cpp - Emit LLVM Code for builtins ---------------------===//
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
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/TargetParser/X86TargetParser.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static std::optional<CodeGenFunction::MSVCIntrin>
translateX86ToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CodeGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::X86::BI_BitScanForward:
  case clang::X86::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::X86::BI_BitScanReverse:
  case clang::X86::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::X86::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::X86::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::X86::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::X86::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::X86::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::X86::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::X86::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::X86::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::X86::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  }
  llvm_unreachable("must return from switch");
}

// Convert the mask from an integer type to a vector of i1.
static Value *getMaskVecValue(CodeGenFunction &CGF, Value *Mask,
                              unsigned NumElts) {

  auto *MaskTy = llvm::FixedVectorType::get(
      CGF.Builder.getInt1Ty(),
      cast<IntegerType>(Mask->getType())->getBitWidth());
  Value *MaskVec = CGF.Builder.CreateBitCast(Mask, MaskTy);

  // If we have less than 8 elements, then the starting mask was an i8 and
  // we need to extract down to the right number of elements.
  if (NumElts < 8) {
    int Indices[4];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    MaskVec = CGF.Builder.CreateShuffleVector(
        MaskVec, MaskVec, ArrayRef(Indices, NumElts), "extract");
  }
  return MaskVec;
}

static Value *EmitX86MaskedStore(CodeGenFunction &CGF, ArrayRef<Value *> Ops,
                                 Align Alignment) {
  Value *Ptr = Ops[0];

  Value *MaskVec = getMaskVecValue(
      CGF, Ops[2],
      cast<llvm::FixedVectorType>(Ops[1]->getType())->getNumElements());

  return CGF.Builder.CreateMaskedStore(Ops[1], Ptr, Alignment, MaskVec);
}

static Value *EmitX86MaskedLoad(CodeGenFunction &CGF, ArrayRef<Value *> Ops,
                                Align Alignment) {
  llvm::Type *Ty = Ops[1]->getType();
  Value *Ptr = Ops[0];

  Value *MaskVec = getMaskVecValue(
      CGF, Ops[2], cast<llvm::FixedVectorType>(Ty)->getNumElements());

  return CGF.Builder.CreateMaskedLoad(Ty, Ptr, Alignment, MaskVec, Ops[1]);
}

static Value *EmitX86ExpandLoad(CodeGenFunction &CGF,
                                ArrayRef<Value *> Ops) {
  auto *ResultTy = cast<llvm::VectorType>(Ops[1]->getType());
  Value *Ptr = Ops[0];

  Value *MaskVec = getMaskVecValue(
      CGF, Ops[2], cast<FixedVectorType>(ResultTy)->getNumElements());

  llvm::Function *F = CGF.CGM.getIntrinsic(Intrinsic::masked_expandload,
                                           ResultTy);
  return CGF.Builder.CreateCall(F, { Ptr, MaskVec, Ops[1] });
}

static Value *EmitX86CompressExpand(CodeGenFunction &CGF,
                                    ArrayRef<Value *> Ops,
                                    bool IsCompress) {
  auto *ResultTy = cast<llvm::FixedVectorType>(Ops[1]->getType());

  Value *MaskVec = getMaskVecValue(CGF, Ops[2], ResultTy->getNumElements());

  Intrinsic::ID IID = IsCompress ? Intrinsic::x86_avx512_mask_compress
                                 : Intrinsic::x86_avx512_mask_expand;
  llvm::Function *F = CGF.CGM.getIntrinsic(IID, ResultTy);
  return CGF.Builder.CreateCall(F, { Ops[0], Ops[1], MaskVec });
}

static Value *EmitX86CompressStore(CodeGenFunction &CGF,
                                   ArrayRef<Value *> Ops) {
  auto *ResultTy = cast<llvm::FixedVectorType>(Ops[1]->getType());
  Value *Ptr = Ops[0];

  Value *MaskVec = getMaskVecValue(CGF, Ops[2], ResultTy->getNumElements());

  llvm::Function *F = CGF.CGM.getIntrinsic(Intrinsic::masked_compressstore,
                                           ResultTy);
  return CGF.Builder.CreateCall(F, { Ops[1], Ptr, MaskVec });
}

static Value *EmitX86MaskLogic(CodeGenFunction &CGF, Instruction::BinaryOps Opc,
                              ArrayRef<Value *> Ops,
                              bool InvertLHS = false) {
  unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
  Value *LHS = getMaskVecValue(CGF, Ops[0], NumElts);
  Value *RHS = getMaskVecValue(CGF, Ops[1], NumElts);

  if (InvertLHS)
    LHS = CGF.Builder.CreateNot(LHS);

  return CGF.Builder.CreateBitCast(CGF.Builder.CreateBinOp(Opc, LHS, RHS),
                                   Ops[0]->getType());
}

static Value *EmitX86FunnelShift(CodeGenFunction &CGF, Value *Op0, Value *Op1,
                                 Value *Amt, bool IsRight) {
  llvm::Type *Ty = Op0->getType();

  // Amount may be scalar immediate, in which case create a splat vector.
  // Funnel shifts amounts are treated as modulo and types are all power-of-2 so
  // we only care about the lowest log2 bits anyway.
  if (Amt->getType() != Ty) {
    unsigned NumElts = cast<llvm::FixedVectorType>(Ty)->getNumElements();
    Amt = CGF.Builder.CreateIntCast(Amt, Ty->getScalarType(), false);
    Amt = CGF.Builder.CreateVectorSplat(NumElts, Amt);
  }

  unsigned IID = IsRight ? Intrinsic::fshr : Intrinsic::fshl;
  Function *F = CGF.CGM.getIntrinsic(IID, Ty);
  return CGF.Builder.CreateCall(F, {Op0, Op1, Amt});
}

static Value *EmitX86vpcom(CodeGenFunction &CGF, ArrayRef<Value *> Ops,
                           bool IsSigned) {
  Value *Op0 = Ops[0];
  Value *Op1 = Ops[1];
  llvm::Type *Ty = Op0->getType();
  uint64_t Imm = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0x7;

  CmpInst::Predicate Pred;
  switch (Imm) {
  case 0x0:
    Pred = IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
    break;
  case 0x1:
    Pred = IsSigned ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE;
    break;
  case 0x2:
    Pred = IsSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
    break;
  case 0x3:
    Pred = IsSigned ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
    break;
  case 0x4:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case 0x5:
    Pred = ICmpInst::ICMP_NE;
    break;
  case 0x6:
    return llvm::Constant::getNullValue(Ty); // FALSE
  case 0x7:
    return llvm::Constant::getAllOnesValue(Ty); // TRUE
  default:
    llvm_unreachable("Unexpected XOP vpcom/vpcomu predicate");
  }

  Value *Cmp = CGF.Builder.CreateICmp(Pred, Op0, Op1);
  Value *Res = CGF.Builder.CreateSExt(Cmp, Ty);
  return Res;
}

static Value *EmitX86Select(CodeGenFunction &CGF,
                            Value *Mask, Value *Op0, Value *Op1) {

  // If the mask is all ones just return first argument.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Op0;

  Mask = getMaskVecValue(
      CGF, Mask, cast<llvm::FixedVectorType>(Op0->getType())->getNumElements());

  return CGF.Builder.CreateSelect(Mask, Op0, Op1);
}

static Value *EmitX86ScalarSelect(CodeGenFunction &CGF,
                                  Value *Mask, Value *Op0, Value *Op1) {
  // If the mask is all ones just return first argument.
  if (const auto *C = dyn_cast<Constant>(Mask))
    if (C->isAllOnesValue())
      return Op0;

  auto *MaskTy = llvm::FixedVectorType::get(
      CGF.Builder.getInt1Ty(), Mask->getType()->getIntegerBitWidth());
  Mask = CGF.Builder.CreateBitCast(Mask, MaskTy);
  Mask = CGF.Builder.CreateExtractElement(Mask, (uint64_t)0);
  return CGF.Builder.CreateSelect(Mask, Op0, Op1);
}

static Value *EmitX86MaskedCompareResult(CodeGenFunction &CGF, Value *Cmp,
                                         unsigned NumElts, Value *MaskIn) {
  if (MaskIn) {
    const auto *C = dyn_cast<Constant>(MaskIn);
    if (!C || !C->isAllOnesValue())
      Cmp = CGF.Builder.CreateAnd(Cmp, getMaskVecValue(CGF, MaskIn, NumElts));
  }

  if (NumElts < 8) {
    int Indices[8];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;
    for (unsigned i = NumElts; i != 8; ++i)
      Indices[i] = i % NumElts + NumElts;
    Cmp = CGF.Builder.CreateShuffleVector(
        Cmp, llvm::Constant::getNullValue(Cmp->getType()), Indices);
  }

  return CGF.Builder.CreateBitCast(Cmp,
                                   IntegerType::get(CGF.getLLVMContext(),
                                                    std::max(NumElts, 8U)));
}

static Value *EmitX86MaskedCompare(CodeGenFunction &CGF, unsigned CC,
                                   bool Signed, ArrayRef<Value *> Ops) {
  assert((Ops.size() == 2 || Ops.size() == 4) &&
         "Unexpected number of arguments");
  unsigned NumElts =
      cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
  Value *Cmp;

  if (CC == 3) {
    Cmp = Constant::getNullValue(
        llvm::FixedVectorType::get(CGF.Builder.getInt1Ty(), NumElts));
  } else if (CC == 7) {
    Cmp = Constant::getAllOnesValue(
        llvm::FixedVectorType::get(CGF.Builder.getInt1Ty(), NumElts));
  } else {
    ICmpInst::Predicate Pred;
    switch (CC) {
    default: llvm_unreachable("Unknown condition code");
    case 0: Pred = ICmpInst::ICMP_EQ;  break;
    case 1: Pred = Signed ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT; break;
    case 2: Pred = Signed ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE; break;
    case 4: Pred = ICmpInst::ICMP_NE;  break;
    case 5: Pred = Signed ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE; break;
    case 6: Pred = Signed ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT; break;
    }
    Cmp = CGF.Builder.CreateICmp(Pred, Ops[0], Ops[1]);
  }

  Value *MaskIn = nullptr;
  if (Ops.size() == 4)
    MaskIn = Ops[3];

  return EmitX86MaskedCompareResult(CGF, Cmp, NumElts, MaskIn);
}

static Value *EmitX86ConvertToMask(CodeGenFunction &CGF, Value *In) {
  Value *Zero = Constant::getNullValue(In->getType());
  return EmitX86MaskedCompare(CGF, 1, true, { In, Zero });
}

static Value *EmitX86ConvertIntToFp(CodeGenFunction &CGF, const CallExpr *E,
                                    ArrayRef<Value *> Ops, bool IsSigned) {
  unsigned Rnd = cast<llvm::ConstantInt>(Ops[3])->getZExtValue();
  llvm::Type *Ty = Ops[1]->getType();

  Value *Res;
  if (Rnd != 4) {
    Intrinsic::ID IID = IsSigned ? Intrinsic::x86_avx512_sitofp_round
                                 : Intrinsic::x86_avx512_uitofp_round;
    Function *F = CGF.CGM.getIntrinsic(IID, { Ty, Ops[0]->getType() });
    Res = CGF.Builder.CreateCall(F, { Ops[0], Ops[3] });
  } else {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
    Res = IsSigned ? CGF.Builder.CreateSIToFP(Ops[0], Ty)
                   : CGF.Builder.CreateUIToFP(Ops[0], Ty);
  }

  return EmitX86Select(CGF, Ops[2], Res, Ops[1]);
}

// Lowers X86 FMA intrinsics to IR.
static Value *EmitX86FMAExpr(CodeGenFunction &CGF, const CallExpr *E,
                             ArrayRef<Value *> Ops, unsigned BuiltinID,
                             bool IsAddSub) {

  bool Subtract = false;
  Intrinsic::ID IID = Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  default: break;
  case clang::X86::BI__builtin_ia32_vfmsubph512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddph512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddph512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddph512_mask3:
    IID = Intrinsic::x86_avx512fp16_vfmadd_ph_512;
    break;
  case clang::X86::BI__builtin_ia32_vfmsubaddph512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_mask3:
    IID = Intrinsic::x86_avx512fp16_vfmaddsub_ph_512;
    break;
  case clang::X86::BI__builtin_ia32_vfmsubps512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddps512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddps512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddps512_mask3:
    IID = Intrinsic::x86_avx512_vfmadd_ps_512; break;
  case clang::X86::BI__builtin_ia32_vfmsubpd512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddpd512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddpd512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddpd512_mask3:
    IID = Intrinsic::x86_avx512_vfmadd_pd_512; break;
  case clang::X86::BI__builtin_ia32_vfmsubaddps512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_mask3:
    IID = Intrinsic::x86_avx512_vfmaddsub_ps_512;
    break;
  case clang::X86::BI__builtin_ia32_vfmsubaddpd512_mask3:
    Subtract = true;
    [[fallthrough]];
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_mask3:
    IID = Intrinsic::x86_avx512_vfmaddsub_pd_512;
    break;
  }

  Value *A = Ops[0];
  Value *B = Ops[1];
  Value *C = Ops[2];

  if (Subtract)
    C = CGF.Builder.CreateFNeg(C);

  Value *Res;

  // Only handle in case of _MM_FROUND_CUR_DIRECTION/4 (no rounding).
  if (IID != Intrinsic::not_intrinsic &&
      (cast<llvm::ConstantInt>(Ops.back())->getZExtValue() != (uint64_t)4 ||
       IsAddSub)) {
    Function *Intr = CGF.CGM.getIntrinsic(IID);
    Res = CGF.Builder.CreateCall(Intr, {A, B, C, Ops.back() });
  } else {
    llvm::Type *Ty = A->getType();
    Function *FMA;
    if (CGF.Builder.getIsFPConstrained()) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
      FMA = CGF.CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, Ty);
      Res = CGF.Builder.CreateConstrainedFPCall(FMA, {A, B, C});
    } else {
      FMA = CGF.CGM.getIntrinsic(Intrinsic::fma, Ty);
      Res = CGF.Builder.CreateCall(FMA, {A, B, C});
    }
  }

  // Handle any required masking.
  Value *MaskFalseVal = nullptr;
  switch (BuiltinID) {
  case clang::X86::BI__builtin_ia32_vfmaddph512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddps512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddpd512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_mask:
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_mask:
    MaskFalseVal = Ops[0];
    break;
  case clang::X86::BI__builtin_ia32_vfmaddph512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddps512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddpd512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_maskz:
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_maskz:
    MaskFalseVal = Constant::getNullValue(Ops[0]->getType());
    break;
  case clang::X86::BI__builtin_ia32_vfmsubph512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddph512_mask3:
  case clang::X86::BI__builtin_ia32_vfmsubps512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddps512_mask3:
  case clang::X86::BI__builtin_ia32_vfmsubpd512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddpd512_mask3:
  case clang::X86::BI__builtin_ia32_vfmsubaddph512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddsubph512_mask3:
  case clang::X86::BI__builtin_ia32_vfmsubaddps512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddsubps512_mask3:
  case clang::X86::BI__builtin_ia32_vfmsubaddpd512_mask3:
  case clang::X86::BI__builtin_ia32_vfmaddsubpd512_mask3:
    MaskFalseVal = Ops[2];
    break;
  }

  if (MaskFalseVal)
    return EmitX86Select(CGF, Ops[3], Res, MaskFalseVal);

  return Res;
}

static Value *EmitScalarFMAExpr(CodeGenFunction &CGF, const CallExpr *E,
                                MutableArrayRef<Value *> Ops, Value *Upper,
                                bool ZeroMask = false, unsigned PTIdx = 0,
                                bool NegAcc = false) {
  unsigned Rnd = 4;
  if (Ops.size() > 4)
    Rnd = cast<llvm::ConstantInt>(Ops[4])->getZExtValue();

  if (NegAcc)
    Ops[2] = CGF.Builder.CreateFNeg(Ops[2]);

  Ops[0] = CGF.Builder.CreateExtractElement(Ops[0], (uint64_t)0);
  Ops[1] = CGF.Builder.CreateExtractElement(Ops[1], (uint64_t)0);
  Ops[2] = CGF.Builder.CreateExtractElement(Ops[2], (uint64_t)0);
  Value *Res;
  if (Rnd != 4) {
    Intrinsic::ID IID;

    switch (Ops[0]->getType()->getPrimitiveSizeInBits()) {
    case 16:
      IID = Intrinsic::x86_avx512fp16_vfmadd_f16;
      break;
    case 32:
      IID = Intrinsic::x86_avx512_vfmadd_f32;
      break;
    case 64:
      IID = Intrinsic::x86_avx512_vfmadd_f64;
      break;
    default:
      llvm_unreachable("Unexpected size");
    }
    Res = CGF.Builder.CreateCall(CGF.CGM.getIntrinsic(IID),
                                 {Ops[0], Ops[1], Ops[2], Ops[4]});
  } else if (CGF.Builder.getIsFPConstrained()) {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
    Function *FMA = CGF.CGM.getIntrinsic(
        Intrinsic::experimental_constrained_fma, Ops[0]->getType());
    Res = CGF.Builder.CreateConstrainedFPCall(FMA, Ops.slice(0, 3));
  } else {
    Function *FMA = CGF.CGM.getIntrinsic(Intrinsic::fma, Ops[0]->getType());
    Res = CGF.Builder.CreateCall(FMA, Ops.slice(0, 3));
  }
  // If we have more than 3 arguments, we need to do masking.
  if (Ops.size() > 3) {
    Value *PassThru = ZeroMask ? Constant::getNullValue(Res->getType())
                               : Ops[PTIdx];

    // If we negated the accumulator and the its the PassThru value we need to
    // bypass the negate. Conveniently Upper should be the same thing in this
    // case.
    if (NegAcc && PTIdx == 2)
      PassThru = CGF.Builder.CreateExtractElement(Upper, (uint64_t)0);

    Res = EmitX86ScalarSelect(CGF, Ops[3], Res, PassThru);
  }
  return CGF.Builder.CreateInsertElement(Upper, Res, (uint64_t)0);
}

static Value *EmitX86Muldq(CodeGenFunction &CGF, bool IsSigned,
                           ArrayRef<Value *> Ops) {
  llvm::Type *Ty = Ops[0]->getType();
  // Arguments have a vXi32 type so cast to vXi64.
  Ty = llvm::FixedVectorType::get(CGF.Int64Ty,
                                  Ty->getPrimitiveSizeInBits() / 64);
  Value *LHS = CGF.Builder.CreateBitCast(Ops[0], Ty);
  Value *RHS = CGF.Builder.CreateBitCast(Ops[1], Ty);

  if (IsSigned) {
    // Shift left then arithmetic shift right.
    Constant *ShiftAmt = ConstantInt::get(Ty, 32);
    LHS = CGF.Builder.CreateShl(LHS, ShiftAmt);
    LHS = CGF.Builder.CreateAShr(LHS, ShiftAmt);
    RHS = CGF.Builder.CreateShl(RHS, ShiftAmt);
    RHS = CGF.Builder.CreateAShr(RHS, ShiftAmt);
  } else {
    // Clear the upper bits.
    Constant *Mask = ConstantInt::get(Ty, 0xffffffff);
    LHS = CGF.Builder.CreateAnd(LHS, Mask);
    RHS = CGF.Builder.CreateAnd(RHS, Mask);
  }

  return CGF.Builder.CreateMul(LHS, RHS);
}

// Emit a masked pternlog intrinsic. This only exists because the header has to
// use a macro and we aren't able to pass the input argument to a pternlog
// builtin and a select builtin without evaluating it twice.
static Value *EmitX86Ternlog(CodeGenFunction &CGF, bool ZeroMask,
                             ArrayRef<Value *> Ops) {
  llvm::Type *Ty = Ops[0]->getType();

  unsigned VecWidth = Ty->getPrimitiveSizeInBits();
  unsigned EltWidth = Ty->getScalarSizeInBits();
  Intrinsic::ID IID;
  if (VecWidth == 128 && EltWidth == 32)
    IID = Intrinsic::x86_avx512_pternlog_d_128;
  else if (VecWidth == 256 && EltWidth == 32)
    IID = Intrinsic::x86_avx512_pternlog_d_256;
  else if (VecWidth == 512 && EltWidth == 32)
    IID = Intrinsic::x86_avx512_pternlog_d_512;
  else if (VecWidth == 128 && EltWidth == 64)
    IID = Intrinsic::x86_avx512_pternlog_q_128;
  else if (VecWidth == 256 && EltWidth == 64)
    IID = Intrinsic::x86_avx512_pternlog_q_256;
  else if (VecWidth == 512 && EltWidth == 64)
    IID = Intrinsic::x86_avx512_pternlog_q_512;
  else
    llvm_unreachable("Unexpected intrinsic");

  Value *Ternlog = CGF.Builder.CreateCall(CGF.CGM.getIntrinsic(IID),
                                          Ops.drop_back());
  Value *PassThru = ZeroMask ? ConstantAggregateZero::get(Ty) : Ops[0];
  return EmitX86Select(CGF, Ops[4], Ternlog, PassThru);
}

static Value *EmitX86SExtMask(CodeGenFunction &CGF, Value *Op,
                              llvm::Type *DstTy) {
  unsigned NumberOfElements =
      cast<llvm::FixedVectorType>(DstTy)->getNumElements();
  Value *Mask = getMaskVecValue(CGF, Op, NumberOfElements);
  return CGF.Builder.CreateSExt(Mask, DstTy, "vpmovm2");
}

Value *CodeGenFunction::EmitX86CpuIs(const CallExpr *E) {
  const Expr *CPUExpr = E->getArg(0)->IgnoreParenCasts();
  StringRef CPUStr = cast<clang::StringLiteral>(CPUExpr)->getString();
  return EmitX86CpuIs(CPUStr);
}

// Convert F16 halfs to floats.
static Value *EmitX86CvtF16ToFloatExpr(CodeGenFunction &CGF,
                                       ArrayRef<Value *> Ops,
                                       llvm::Type *DstTy) {
  assert((Ops.size() == 1 || Ops.size() == 3 || Ops.size() == 4) &&
         "Unknown cvtph2ps intrinsic");

  // If the SAE intrinsic doesn't use default rounding then we can't upgrade.
  if (Ops.size() == 4 && cast<llvm::ConstantInt>(Ops[3])->getZExtValue() != 4) {
    Function *F =
        CGF.CGM.getIntrinsic(Intrinsic::x86_avx512_mask_vcvtph2ps_512);
    return CGF.Builder.CreateCall(F, {Ops[0], Ops[1], Ops[2], Ops[3]});
  }

  unsigned NumDstElts = cast<llvm::FixedVectorType>(DstTy)->getNumElements();
  Value *Src = Ops[0];

  // Extract the subvector.
  if (NumDstElts !=
      cast<llvm::FixedVectorType>(Src->getType())->getNumElements()) {
    assert(NumDstElts == 4 && "Unexpected vector size");
    Src = CGF.Builder.CreateShuffleVector(Src, {0, 1, 2, 3});
  }

  // Bitcast from vXi16 to vXf16.
  auto *HalfTy = llvm::FixedVectorType::get(
      llvm::Type::getHalfTy(CGF.getLLVMContext()), NumDstElts);
  Src = CGF.Builder.CreateBitCast(Src, HalfTy);

  // Perform the fp-extension.
  Value *Res = CGF.Builder.CreateFPExt(Src, DstTy, "cvtph2ps");

  if (Ops.size() >= 3)
    Res = EmitX86Select(CGF, Ops[2], Res, Ops[1]);
  return Res;
}

Value *CodeGenFunction::EmitX86CpuIs(StringRef CPUStr) {

  llvm::Type *Int32Ty = Builder.getInt32Ty();

  // Matching the struct layout from the compiler-rt/libgcc structure that is
  // filled in:
  // unsigned int __cpu_vendor;
  // unsigned int __cpu_type;
  // unsigned int __cpu_subtype;
  // unsigned int __cpu_features[1];
  llvm::Type *STy = llvm::StructType::get(Int32Ty, Int32Ty, Int32Ty,
                                          llvm::ArrayType::get(Int32Ty, 1));

  // Grab the global __cpu_model.
  llvm::Constant *CpuModel = CGM.CreateRuntimeVariable(STy, "__cpu_model");
  cast<llvm::GlobalValue>(CpuModel)->setDSOLocal(true);

  // Calculate the index needed to access the correct field based on the
  // range. Also adjust the expected value.
  auto [Index, Value] = StringSwitch<std::pair<unsigned, unsigned>>(CPUStr)
#define X86_VENDOR(ENUM, STRING)                                               \
  .Case(STRING, {0u, static_cast<unsigned>(llvm::X86::ENUM)})
#define X86_CPU_TYPE_ALIAS(ENUM, ALIAS)                                        \
  .Case(ALIAS, {1u, static_cast<unsigned>(llvm::X86::ENUM)})
#define X86_CPU_TYPE(ENUM, STR)                                                \
  .Case(STR, {1u, static_cast<unsigned>(llvm::X86::ENUM)})
#define X86_CPU_SUBTYPE_ALIAS(ENUM, ALIAS)                                     \
  .Case(ALIAS, {2u, static_cast<unsigned>(llvm::X86::ENUM)})
#define X86_CPU_SUBTYPE(ENUM, STR)                                             \
  .Case(STR, {2u, static_cast<unsigned>(llvm::X86::ENUM)})
#include "llvm/TargetParser/X86TargetParser.def"
                               .Default({0, 0});
  assert(Value != 0 && "Invalid CPUStr passed to CpuIs");

  // Grab the appropriate field from __cpu_model.
  llvm::Value *Idxs[] = {ConstantInt::get(Int32Ty, 0),
                         ConstantInt::get(Int32Ty, Index)};
  llvm::Value *CpuValue = Builder.CreateInBoundsGEP(STy, CpuModel, Idxs);
  CpuValue = Builder.CreateAlignedLoad(Int32Ty, CpuValue,
                                       CharUnits::fromQuantity(4));

  // Check the value of the field against the requested value.
  return Builder.CreateICmpEQ(CpuValue,
                                  llvm::ConstantInt::get(Int32Ty, Value));
}

Value *CodeGenFunction::EmitX86CpuSupports(const CallExpr *E) {
  const Expr *FeatureExpr = E->getArg(0)->IgnoreParenCasts();
  StringRef FeatureStr = cast<StringLiteral>(FeatureExpr)->getString();
  if (!getContext().getTargetInfo().validateCpuSupports(FeatureStr))
    return Builder.getFalse();
  return EmitX86CpuSupports(FeatureStr);
}

Value *CodeGenFunction::EmitX86CpuSupports(ArrayRef<StringRef> FeatureStrs) {
  return EmitX86CpuSupports(llvm::X86::getCpuSupportsMask(FeatureStrs));
}

llvm::Value *
CodeGenFunction::EmitX86CpuSupports(std::array<uint32_t, 4> FeatureMask) {
  Value *Result = Builder.getTrue();
  if (FeatureMask[0] != 0) {
    // Matching the struct layout from the compiler-rt/libgcc structure that is
    // filled in:
    // unsigned int __cpu_vendor;
    // unsigned int __cpu_type;
    // unsigned int __cpu_subtype;
    // unsigned int __cpu_features[1];
    llvm::Type *STy = llvm::StructType::get(Int32Ty, Int32Ty, Int32Ty,
                                            llvm::ArrayType::get(Int32Ty, 1));

    // Grab the global __cpu_model.
    llvm::Constant *CpuModel = CGM.CreateRuntimeVariable(STy, "__cpu_model");
    cast<llvm::GlobalValue>(CpuModel)->setDSOLocal(true);

    // Grab the first (0th) element from the field __cpu_features off of the
    // global in the struct STy.
    Value *Idxs[] = {Builder.getInt32(0), Builder.getInt32(3),
                     Builder.getInt32(0)};
    Value *CpuFeatures = Builder.CreateInBoundsGEP(STy, CpuModel, Idxs);
    Value *Features = Builder.CreateAlignedLoad(Int32Ty, CpuFeatures,
                                                CharUnits::fromQuantity(4));

    // Check the value of the bit corresponding to the feature requested.
    Value *Mask = Builder.getInt32(FeatureMask[0]);
    Value *Bitset = Builder.CreateAnd(Features, Mask);
    Value *Cmp = Builder.CreateICmpEQ(Bitset, Mask);
    Result = Builder.CreateAnd(Result, Cmp);
  }

  llvm::Type *ATy = llvm::ArrayType::get(Int32Ty, 3);
  llvm::Constant *CpuFeatures2 =
      CGM.CreateRuntimeVariable(ATy, "__cpu_features2");
  cast<llvm::GlobalValue>(CpuFeatures2)->setDSOLocal(true);
  for (int i = 1; i != 4; ++i) {
    const uint32_t M = FeatureMask[i];
    if (!M)
      continue;
    Value *Idxs[] = {Builder.getInt32(0), Builder.getInt32(i - 1)};
    Value *Features = Builder.CreateAlignedLoad(
        Int32Ty, Builder.CreateInBoundsGEP(ATy, CpuFeatures2, Idxs),
        CharUnits::fromQuantity(4));
    // Check the value of the bit corresponding to the feature requested.
    Value *Mask = Builder.getInt32(M);
    Value *Bitset = Builder.CreateAnd(Features, Mask);
    Value *Cmp = Builder.CreateICmpEQ(Bitset, Mask);
    Result = Builder.CreateAnd(Result, Cmp);
  }

  return Result;
}

Value *CodeGenFunction::EmitX86CpuInit() {
  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy,
                                                    /*Variadic*/ false);
  llvm::FunctionCallee Func =
      CGM.CreateRuntimeFunction(FTy, "__cpu_indicator_init");
  cast<llvm::GlobalValue>(Func.getCallee())->setDSOLocal(true);
  cast<llvm::GlobalValue>(Func.getCallee())
      ->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
  return Builder.CreateCall(Func);
}


Value *CodeGenFunction::EmitX86BuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  if (BuiltinID == Builtin::BI__builtin_cpu_is)
    return EmitX86CpuIs(E);
  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    return EmitX86CpuSupports(E);
  if (BuiltinID == Builtin::BI__builtin_cpu_init)
    return EmitX86CpuInit();

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<MSVCIntrin> MsvcIntId = translateX86ToMsvcIntrin(BuiltinID))
    return EmitMSVCBuiltinExpr(*MsvcIntId, E);

  SmallVector<Value*, 4> Ops;
  bool IsMaskFCmp = false;
  bool IsConjFMA = false;

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    Ops.push_back(EmitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  // These exist so that the builtin that takes an immediate can be bounds
  // checked by clang to avoid passing bad immediates to the backend. Since
  // AVX has a larger immediate than SSE we would need separate builtins to
  // do the different bounds checking. Rather than create a clang specific
  // SSE only builtin, this implements eight separate builtins to match gcc
  // implementation.
  auto getCmpIntrinsicCall = [this, &Ops](Intrinsic::ID ID, unsigned Imm) {
    Ops.push_back(llvm::ConstantInt::get(Int8Ty, Imm));
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops);
  };

  // For the vector forms of FP comparisons, translate the builtins directly to
  // IR.
  // TODO: The builtins could be removed if the SSE header files used vector
  // extension comparisons directly (vector ordered/unordered may need
  // additional support via __builtin_isnan()).
  auto getVectorFCmpIR = [this, &Ops, E](CmpInst::Predicate Pred,
                                         bool IsSignaling) {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *Cmp;
    if (IsSignaling)
      Cmp = Builder.CreateFCmpS(Pred, Ops[0], Ops[1]);
    else
      Cmp = Builder.CreateFCmp(Pred, Ops[0], Ops[1]);
    llvm::VectorType *FPVecTy = cast<llvm::VectorType>(Ops[0]->getType());
    llvm::VectorType *IntVecTy = llvm::VectorType::getInteger(FPVecTy);
    Value *Sext = Builder.CreateSExt(Cmp, IntVecTy);
    return Builder.CreateBitCast(Sext, FPVecTy);
  };

  switch (BuiltinID) {
  default: return nullptr;
  case X86::BI_mm_prefetch: {
    Value *Address = Ops[0];
    ConstantInt *C = cast<ConstantInt>(Ops[1]);
    Value *RW = ConstantInt::get(Int32Ty, (C->getZExtValue() >> 2) & 0x1);
    Value *Locality = ConstantInt::get(Int32Ty, C->getZExtValue() & 0x3);
    Value *Data = ConstantInt::get(Int32Ty, 1);
    Function *F = CGM.getIntrinsic(Intrinsic::prefetch, Address->getType());
    return Builder.CreateCall(F, {Address, RW, Locality, Data});
  }
  case X86::BI_m_prefetch:
  case X86::BI_m_prefetchw: {
    Value *Address = Ops[0];
    // The 'w' suffix implies write.
    Value *RW =
        ConstantInt::get(Int32Ty, BuiltinID == X86::BI_m_prefetchw ? 1 : 0);
    Value *Locality = ConstantInt::get(Int32Ty, 0x3);
    Value *Data = ConstantInt::get(Int32Ty, 1);
    Function *F = CGM.getIntrinsic(Intrinsic::prefetch, Address->getType());
    return Builder.CreateCall(F, {Address, RW, Locality, Data});
  }
  case X86::BI_mm_clflush: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse2_clflush),
                              Ops[0]);
  }
  case X86::BI_mm_lfence: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse2_lfence));
  }
  case X86::BI_mm_mfence: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse2_mfence));
  }
  case X86::BI_mm_sfence: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_sfence));
  }
  case X86::BI_mm_pause: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse2_pause));
  }
  case X86::BI__rdtsc: {
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_rdtsc));
  }
  case X86::BI__builtin_ia32_rdtscp: {
    Value *Call = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_rdtscp));
    Builder.CreateDefaultAlignedStore(Builder.CreateExtractValue(Call, 1),
                                      Ops[0]);
    return Builder.CreateExtractValue(Call, 0);
  }
  case X86::BI__builtin_ia32_lzcnt_u16:
  case X86::BI__builtin_ia32_lzcnt_u32:
  case X86::BI__builtin_ia32_lzcnt_u64: {
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Ops[0]->getType());
    return Builder.CreateCall(F, {Ops[0], Builder.getInt1(false)});
  }
  case X86::BI__builtin_ia32_tzcnt_u16:
  case X86::BI__builtin_ia32_tzcnt_u32:
  case X86::BI__builtin_ia32_tzcnt_u64: {
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, Ops[0]->getType());
    return Builder.CreateCall(F, {Ops[0], Builder.getInt1(false)});
  }
  case X86::BI__builtin_ia32_undef128:
  case X86::BI__builtin_ia32_undef256:
  case X86::BI__builtin_ia32_undef512:
    // The x86 definition of "undef" is not the same as the LLVM definition
    // (PR32176). We leave optimizing away an unnecessary zero constant to the
    // IR optimizer and backend.
    // TODO: If we had a "freeze" IR instruction to generate a fixed undef
    // value, we should use that here instead of a zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  case X86::BI__builtin_ia32_vec_ext_v4hi:
  case X86::BI__builtin_ia32_vec_ext_v16qi:
  case X86::BI__builtin_ia32_vec_ext_v8hi:
  case X86::BI__builtin_ia32_vec_ext_v4si:
  case X86::BI__builtin_ia32_vec_ext_v4sf:
  case X86::BI__builtin_ia32_vec_ext_v2di:
  case X86::BI__builtin_ia32_vec_ext_v32qi:
  case X86::BI__builtin_ia32_vec_ext_v16hi:
  case X86::BI__builtin_ia32_vec_ext_v8si:
  case X86::BI__builtin_ia32_vec_ext_v4di: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    uint64_t Index = cast<ConstantInt>(Ops[1])->getZExtValue();
    Index &= NumElts - 1;
    // These builtins exist so we can ensure the index is an ICE and in range.
    // Otherwise we could just do this in the header file.
    return Builder.CreateExtractElement(Ops[0], Index);
  }
  case X86::BI__builtin_ia32_vec_set_v4hi:
  case X86::BI__builtin_ia32_vec_set_v16qi:
  case X86::BI__builtin_ia32_vec_set_v8hi:
  case X86::BI__builtin_ia32_vec_set_v4si:
  case X86::BI__builtin_ia32_vec_set_v2di:
  case X86::BI__builtin_ia32_vec_set_v32qi:
  case X86::BI__builtin_ia32_vec_set_v16hi:
  case X86::BI__builtin_ia32_vec_set_v8si:
  case X86::BI__builtin_ia32_vec_set_v4di: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    unsigned Index = cast<ConstantInt>(Ops[2])->getZExtValue();
    Index &= NumElts - 1;
    // These builtins exist so we can ensure the index is an ICE and in range.
    // Otherwise we could just do this in the header file.
    return Builder.CreateInsertElement(Ops[0], Ops[1], Index);
  }
  case X86::BI_mm_setcsr:
  case X86::BI__builtin_ia32_ldmxcsr: {
    RawAddress Tmp = CreateMemTemp(E->getArg(0)->getType());
    Builder.CreateStore(Ops[0], Tmp);
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_ldmxcsr),
                              Tmp.getPointer());
  }
  case X86::BI_mm_getcsr:
  case X86::BI__builtin_ia32_stmxcsr: {
    RawAddress Tmp = CreateMemTemp(E->getType());
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_stmxcsr),
                       Tmp.getPointer());
    return Builder.CreateLoad(Tmp, "stmxcsr");
  }
  case X86::BI__builtin_ia32_xsave:
  case X86::BI__builtin_ia32_xsave64:
  case X86::BI__builtin_ia32_xrstor:
  case X86::BI__builtin_ia32_xrstor64:
  case X86::BI__builtin_ia32_xsaveopt:
  case X86::BI__builtin_ia32_xsaveopt64:
  case X86::BI__builtin_ia32_xrstors:
  case X86::BI__builtin_ia32_xrstors64:
  case X86::BI__builtin_ia32_xsavec:
  case X86::BI__builtin_ia32_xsavec64:
  case X86::BI__builtin_ia32_xsaves:
  case X86::BI__builtin_ia32_xsaves64:
  case X86::BI__builtin_ia32_xsetbv:
  case X86::BI_xsetbv: {
    Intrinsic::ID ID;
#define INTRINSIC_X86_XSAVE_ID(NAME) \
    case X86::BI__builtin_ia32_##NAME: \
      ID = Intrinsic::x86_##NAME; \
      break
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    INTRINSIC_X86_XSAVE_ID(xsave);
    INTRINSIC_X86_XSAVE_ID(xsave64);
    INTRINSIC_X86_XSAVE_ID(xrstor);
    INTRINSIC_X86_XSAVE_ID(xrstor64);
    INTRINSIC_X86_XSAVE_ID(xsaveopt);
    INTRINSIC_X86_XSAVE_ID(xsaveopt64);
    INTRINSIC_X86_XSAVE_ID(xrstors);
    INTRINSIC_X86_XSAVE_ID(xrstors64);
    INTRINSIC_X86_XSAVE_ID(xsavec);
    INTRINSIC_X86_XSAVE_ID(xsavec64);
    INTRINSIC_X86_XSAVE_ID(xsaves);
    INTRINSIC_X86_XSAVE_ID(xsaves64);
    INTRINSIC_X86_XSAVE_ID(xsetbv);
    case X86::BI_xsetbv:
      ID = Intrinsic::x86_xsetbv;
      break;
    }
#undef INTRINSIC_X86_XSAVE_ID
    Value *Mhi = Builder.CreateTrunc(
      Builder.CreateLShr(Ops[1], ConstantInt::get(Int64Ty, 32)), Int32Ty);
    Value *Mlo = Builder.CreateTrunc(Ops[1], Int32Ty);
    Ops[1] = Mhi;
    Ops.push_back(Mlo);
    return Builder.CreateCall(CGM.getIntrinsic(ID), Ops);
  }
  case X86::BI__builtin_ia32_xgetbv:
  case X86::BI_xgetbv:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_xgetbv), Ops);
  case X86::BI__builtin_ia32_storedqudi128_mask:
  case X86::BI__builtin_ia32_storedqusi128_mask:
  case X86::BI__builtin_ia32_storedquhi128_mask:
  case X86::BI__builtin_ia32_storedquqi128_mask:
  case X86::BI__builtin_ia32_storeupd128_mask:
  case X86::BI__builtin_ia32_storeups128_mask:
  case X86::BI__builtin_ia32_storedqudi256_mask:
  case X86::BI__builtin_ia32_storedqusi256_mask:
  case X86::BI__builtin_ia32_storedquhi256_mask:
  case X86::BI__builtin_ia32_storedquqi256_mask:
  case X86::BI__builtin_ia32_storeupd256_mask:
  case X86::BI__builtin_ia32_storeups256_mask:
  case X86::BI__builtin_ia32_storedqudi512_mask:
  case X86::BI__builtin_ia32_storedqusi512_mask:
  case X86::BI__builtin_ia32_storedquhi512_mask:
  case X86::BI__builtin_ia32_storedquqi512_mask:
  case X86::BI__builtin_ia32_storeupd512_mask:
  case X86::BI__builtin_ia32_storeups512_mask:
    return EmitX86MaskedStore(*this, Ops, Align(1));

  case X86::BI__builtin_ia32_storesbf16128_mask:
  case X86::BI__builtin_ia32_storesh128_mask:
  case X86::BI__builtin_ia32_storess128_mask:
  case X86::BI__builtin_ia32_storesd128_mask:
    return EmitX86MaskedStore(*this, Ops, Align(1));

  case X86::BI__builtin_ia32_cvtmask2b128:
  case X86::BI__builtin_ia32_cvtmask2b256:
  case X86::BI__builtin_ia32_cvtmask2b512:
  case X86::BI__builtin_ia32_cvtmask2w128:
  case X86::BI__builtin_ia32_cvtmask2w256:
  case X86::BI__builtin_ia32_cvtmask2w512:
  case X86::BI__builtin_ia32_cvtmask2d128:
  case X86::BI__builtin_ia32_cvtmask2d256:
  case X86::BI__builtin_ia32_cvtmask2d512:
  case X86::BI__builtin_ia32_cvtmask2q128:
  case X86::BI__builtin_ia32_cvtmask2q256:
  case X86::BI__builtin_ia32_cvtmask2q512:
    return EmitX86SExtMask(*this, Ops[0], ConvertType(E->getType()));

  case X86::BI__builtin_ia32_cvtb2mask128:
  case X86::BI__builtin_ia32_cvtb2mask256:
  case X86::BI__builtin_ia32_cvtb2mask512:
  case X86::BI__builtin_ia32_cvtw2mask128:
  case X86::BI__builtin_ia32_cvtw2mask256:
  case X86::BI__builtin_ia32_cvtw2mask512:
  case X86::BI__builtin_ia32_cvtd2mask128:
  case X86::BI__builtin_ia32_cvtd2mask256:
  case X86::BI__builtin_ia32_cvtd2mask512:
  case X86::BI__builtin_ia32_cvtq2mask128:
  case X86::BI__builtin_ia32_cvtq2mask256:
  case X86::BI__builtin_ia32_cvtq2mask512:
    return EmitX86ConvertToMask(*this, Ops[0]);

  case X86::BI__builtin_ia32_cvtdq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtdq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtqq2ph512_mask:
    return EmitX86ConvertIntToFp(*this, E, Ops, /*IsSigned*/ true);
  case X86::BI__builtin_ia32_cvtudq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtuw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtudq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtuqq2ph512_mask:
    return EmitX86ConvertIntToFp(*this, E, Ops, /*IsSigned*/ false);

  case X86::BI__builtin_ia32_vfmaddss3:
  case X86::BI__builtin_ia32_vfmaddsd3:
  case X86::BI__builtin_ia32_vfmaddsh3_mask:
  case X86::BI__builtin_ia32_vfmaddss3_mask:
  case X86::BI__builtin_ia32_vfmaddsd3_mask:
    return EmitScalarFMAExpr(*this, E, Ops, Ops[0]);
  case X86::BI__builtin_ia32_vfmaddss:
  case X86::BI__builtin_ia32_vfmaddsd:
    return EmitScalarFMAExpr(*this, E, Ops,
                             Constant::getNullValue(Ops[0]->getType()));
  case X86::BI__builtin_ia32_vfmaddsh3_maskz:
  case X86::BI__builtin_ia32_vfmaddss3_maskz:
  case X86::BI__builtin_ia32_vfmaddsd3_maskz:
    return EmitScalarFMAExpr(*this, E, Ops, Ops[0], /*ZeroMask*/ true);
  case X86::BI__builtin_ia32_vfmaddsh3_mask3:
  case X86::BI__builtin_ia32_vfmaddss3_mask3:
  case X86::BI__builtin_ia32_vfmaddsd3_mask3:
    return EmitScalarFMAExpr(*this, E, Ops, Ops[2], /*ZeroMask*/ false, 2);
  case X86::BI__builtin_ia32_vfmsubsh3_mask3:
  case X86::BI__builtin_ia32_vfmsubss3_mask3:
  case X86::BI__builtin_ia32_vfmsubsd3_mask3:
    return EmitScalarFMAExpr(*this, E, Ops, Ops[2], /*ZeroMask*/ false, 2,
                             /*NegAcc*/ true);
  case X86::BI__builtin_ia32_vfmaddph512_mask:
  case X86::BI__builtin_ia32_vfmaddph512_maskz:
  case X86::BI__builtin_ia32_vfmaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddps512_mask:
  case X86::BI__builtin_ia32_vfmaddps512_maskz:
  case X86::BI__builtin_ia32_vfmaddps512_mask3:
  case X86::BI__builtin_ia32_vfmsubps512_mask3:
  case X86::BI__builtin_ia32_vfmaddpd512_mask:
  case X86::BI__builtin_ia32_vfmaddpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubph512_mask3:
    return EmitX86FMAExpr(*this, E, Ops, BuiltinID, /*IsAddSub*/ false);
  case X86::BI__builtin_ia32_vfmaddsubph512_mask:
  case X86::BI__builtin_ia32_vfmaddsubph512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubph512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask:
  case X86::BI__builtin_ia32_vfmaddsubps512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddps512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask:
  case X86::BI__builtin_ia32_vfmaddsubpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddpd512_mask3:
    return EmitX86FMAExpr(*this, E, Ops, BuiltinID, /*IsAddSub*/ true);

  case X86::BI__builtin_ia32_movdqa32store128_mask:
  case X86::BI__builtin_ia32_movdqa64store128_mask:
  case X86::BI__builtin_ia32_storeaps128_mask:
  case X86::BI__builtin_ia32_storeapd128_mask:
  case X86::BI__builtin_ia32_movdqa32store256_mask:
  case X86::BI__builtin_ia32_movdqa64store256_mask:
  case X86::BI__builtin_ia32_storeaps256_mask:
  case X86::BI__builtin_ia32_storeapd256_mask:
  case X86::BI__builtin_ia32_movdqa32store512_mask:
  case X86::BI__builtin_ia32_movdqa64store512_mask:
  case X86::BI__builtin_ia32_storeaps512_mask:
  case X86::BI__builtin_ia32_storeapd512_mask:
    return EmitX86MaskedStore(
        *this, Ops,
        getContext().getTypeAlignInChars(E->getArg(1)->getType()).getAsAlign());

  case X86::BI__builtin_ia32_loadups128_mask:
  case X86::BI__builtin_ia32_loadups256_mask:
  case X86::BI__builtin_ia32_loadups512_mask:
  case X86::BI__builtin_ia32_loadupd128_mask:
  case X86::BI__builtin_ia32_loadupd256_mask:
  case X86::BI__builtin_ia32_loadupd512_mask:
  case X86::BI__builtin_ia32_loaddquqi128_mask:
  case X86::BI__builtin_ia32_loaddquqi256_mask:
  case X86::BI__builtin_ia32_loaddquqi512_mask:
  case X86::BI__builtin_ia32_loaddquhi128_mask:
  case X86::BI__builtin_ia32_loaddquhi256_mask:
  case X86::BI__builtin_ia32_loaddquhi512_mask:
  case X86::BI__builtin_ia32_loaddqusi128_mask:
  case X86::BI__builtin_ia32_loaddqusi256_mask:
  case X86::BI__builtin_ia32_loaddqusi512_mask:
  case X86::BI__builtin_ia32_loaddqudi128_mask:
  case X86::BI__builtin_ia32_loaddqudi256_mask:
  case X86::BI__builtin_ia32_loaddqudi512_mask:
    return EmitX86MaskedLoad(*this, Ops, Align(1));

  case X86::BI__builtin_ia32_loadsbf16128_mask:
  case X86::BI__builtin_ia32_loadsh128_mask:
  case X86::BI__builtin_ia32_loadss128_mask:
  case X86::BI__builtin_ia32_loadsd128_mask:
    return EmitX86MaskedLoad(*this, Ops, Align(1));

  case X86::BI__builtin_ia32_loadaps128_mask:
  case X86::BI__builtin_ia32_loadaps256_mask:
  case X86::BI__builtin_ia32_loadaps512_mask:
  case X86::BI__builtin_ia32_loadapd128_mask:
  case X86::BI__builtin_ia32_loadapd256_mask:
  case X86::BI__builtin_ia32_loadapd512_mask:
  case X86::BI__builtin_ia32_movdqa32load128_mask:
  case X86::BI__builtin_ia32_movdqa32load256_mask:
  case X86::BI__builtin_ia32_movdqa32load512_mask:
  case X86::BI__builtin_ia32_movdqa64load128_mask:
  case X86::BI__builtin_ia32_movdqa64load256_mask:
  case X86::BI__builtin_ia32_movdqa64load512_mask:
    return EmitX86MaskedLoad(
        *this, Ops,
        getContext().getTypeAlignInChars(E->getArg(1)->getType()).getAsAlign());

  case X86::BI__builtin_ia32_expandloaddf128_mask:
  case X86::BI__builtin_ia32_expandloaddf256_mask:
  case X86::BI__builtin_ia32_expandloaddf512_mask:
  case X86::BI__builtin_ia32_expandloadsf128_mask:
  case X86::BI__builtin_ia32_expandloadsf256_mask:
  case X86::BI__builtin_ia32_expandloadsf512_mask:
  case X86::BI__builtin_ia32_expandloaddi128_mask:
  case X86::BI__builtin_ia32_expandloaddi256_mask:
  case X86::BI__builtin_ia32_expandloaddi512_mask:
  case X86::BI__builtin_ia32_expandloadsi128_mask:
  case X86::BI__builtin_ia32_expandloadsi256_mask:
  case X86::BI__builtin_ia32_expandloadsi512_mask:
  case X86::BI__builtin_ia32_expandloadhi128_mask:
  case X86::BI__builtin_ia32_expandloadhi256_mask:
  case X86::BI__builtin_ia32_expandloadhi512_mask:
  case X86::BI__builtin_ia32_expandloadqi128_mask:
  case X86::BI__builtin_ia32_expandloadqi256_mask:
  case X86::BI__builtin_ia32_expandloadqi512_mask:
    return EmitX86ExpandLoad(*this, Ops);

  case X86::BI__builtin_ia32_compressstoredf128_mask:
  case X86::BI__builtin_ia32_compressstoredf256_mask:
  case X86::BI__builtin_ia32_compressstoredf512_mask:
  case X86::BI__builtin_ia32_compressstoresf128_mask:
  case X86::BI__builtin_ia32_compressstoresf256_mask:
  case X86::BI__builtin_ia32_compressstoresf512_mask:
  case X86::BI__builtin_ia32_compressstoredi128_mask:
  case X86::BI__builtin_ia32_compressstoredi256_mask:
  case X86::BI__builtin_ia32_compressstoredi512_mask:
  case X86::BI__builtin_ia32_compressstoresi128_mask:
  case X86::BI__builtin_ia32_compressstoresi256_mask:
  case X86::BI__builtin_ia32_compressstoresi512_mask:
  case X86::BI__builtin_ia32_compressstorehi128_mask:
  case X86::BI__builtin_ia32_compressstorehi256_mask:
  case X86::BI__builtin_ia32_compressstorehi512_mask:
  case X86::BI__builtin_ia32_compressstoreqi128_mask:
  case X86::BI__builtin_ia32_compressstoreqi256_mask:
  case X86::BI__builtin_ia32_compressstoreqi512_mask:
    return EmitX86CompressStore(*this, Ops);

  case X86::BI__builtin_ia32_expanddf128_mask:
  case X86::BI__builtin_ia32_expanddf256_mask:
  case X86::BI__builtin_ia32_expanddf512_mask:
  case X86::BI__builtin_ia32_expandsf128_mask:
  case X86::BI__builtin_ia32_expandsf256_mask:
  case X86::BI__builtin_ia32_expandsf512_mask:
  case X86::BI__builtin_ia32_expanddi128_mask:
  case X86::BI__builtin_ia32_expanddi256_mask:
  case X86::BI__builtin_ia32_expanddi512_mask:
  case X86::BI__builtin_ia32_expandsi128_mask:
  case X86::BI__builtin_ia32_expandsi256_mask:
  case X86::BI__builtin_ia32_expandsi512_mask:
  case X86::BI__builtin_ia32_expandhi128_mask:
  case X86::BI__builtin_ia32_expandhi256_mask:
  case X86::BI__builtin_ia32_expandhi512_mask:
  case X86::BI__builtin_ia32_expandqi128_mask:
  case X86::BI__builtin_ia32_expandqi256_mask:
  case X86::BI__builtin_ia32_expandqi512_mask:
    return EmitX86CompressExpand(*this, Ops, /*IsCompress*/false);

  case X86::BI__builtin_ia32_compressdf128_mask:
  case X86::BI__builtin_ia32_compressdf256_mask:
  case X86::BI__builtin_ia32_compressdf512_mask:
  case X86::BI__builtin_ia32_compresssf128_mask:
  case X86::BI__builtin_ia32_compresssf256_mask:
  case X86::BI__builtin_ia32_compresssf512_mask:
  case X86::BI__builtin_ia32_compressdi128_mask:
  case X86::BI__builtin_ia32_compressdi256_mask:
  case X86::BI__builtin_ia32_compressdi512_mask:
  case X86::BI__builtin_ia32_compresssi128_mask:
  case X86::BI__builtin_ia32_compresssi256_mask:
  case X86::BI__builtin_ia32_compresssi512_mask:
  case X86::BI__builtin_ia32_compresshi128_mask:
  case X86::BI__builtin_ia32_compresshi256_mask:
  case X86::BI__builtin_ia32_compresshi512_mask:
  case X86::BI__builtin_ia32_compressqi128_mask:
  case X86::BI__builtin_ia32_compressqi256_mask:
  case X86::BI__builtin_ia32_compressqi512_mask:
    return EmitX86CompressExpand(*this, Ops, /*IsCompress*/true);

  case X86::BI__builtin_ia32_gather3div2df:
  case X86::BI__builtin_ia32_gather3div2di:
  case X86::BI__builtin_ia32_gather3div4df:
  case X86::BI__builtin_ia32_gather3div4di:
  case X86::BI__builtin_ia32_gather3div4sf:
  case X86::BI__builtin_ia32_gather3div4si:
  case X86::BI__builtin_ia32_gather3div8sf:
  case X86::BI__builtin_ia32_gather3div8si:
  case X86::BI__builtin_ia32_gather3siv2df:
  case X86::BI__builtin_ia32_gather3siv2di:
  case X86::BI__builtin_ia32_gather3siv4df:
  case X86::BI__builtin_ia32_gather3siv4di:
  case X86::BI__builtin_ia32_gather3siv4sf:
  case X86::BI__builtin_ia32_gather3siv4si:
  case X86::BI__builtin_ia32_gather3siv8sf:
  case X86::BI__builtin_ia32_gather3siv8si:
  case X86::BI__builtin_ia32_gathersiv8df:
  case X86::BI__builtin_ia32_gathersiv16sf:
  case X86::BI__builtin_ia32_gatherdiv8df:
  case X86::BI__builtin_ia32_gatherdiv16sf:
  case X86::BI__builtin_ia32_gathersiv8di:
  case X86::BI__builtin_ia32_gathersiv16si:
  case X86::BI__builtin_ia32_gatherdiv8di:
  case X86::BI__builtin_ia32_gatherdiv16si: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_gather3div2df:
      IID = Intrinsic::x86_avx512_mask_gather3div2_df;
      break;
    case X86::BI__builtin_ia32_gather3div2di:
      IID = Intrinsic::x86_avx512_mask_gather3div2_di;
      break;
    case X86::BI__builtin_ia32_gather3div4df:
      IID = Intrinsic::x86_avx512_mask_gather3div4_df;
      break;
    case X86::BI__builtin_ia32_gather3div4di:
      IID = Intrinsic::x86_avx512_mask_gather3div4_di;
      break;
    case X86::BI__builtin_ia32_gather3div4sf:
      IID = Intrinsic::x86_avx512_mask_gather3div4_sf;
      break;
    case X86::BI__builtin_ia32_gather3div4si:
      IID = Intrinsic::x86_avx512_mask_gather3div4_si;
      break;
    case X86::BI__builtin_ia32_gather3div8sf:
      IID = Intrinsic::x86_avx512_mask_gather3div8_sf;
      break;
    case X86::BI__builtin_ia32_gather3div8si:
      IID = Intrinsic::x86_avx512_mask_gather3div8_si;
      break;
    case X86::BI__builtin_ia32_gather3siv2df:
      IID = Intrinsic::x86_avx512_mask_gather3siv2_df;
      break;
    case X86::BI__builtin_ia32_gather3siv2di:
      IID = Intrinsic::x86_avx512_mask_gather3siv2_di;
      break;
    case X86::BI__builtin_ia32_gather3siv4df:
      IID = Intrinsic::x86_avx512_mask_gather3siv4_df;
      break;
    case X86::BI__builtin_ia32_gather3siv4di:
      IID = Intrinsic::x86_avx512_mask_gather3siv4_di;
      break;
    case X86::BI__builtin_ia32_gather3siv4sf:
      IID = Intrinsic::x86_avx512_mask_gather3siv4_sf;
      break;
    case X86::BI__builtin_ia32_gather3siv4si:
      IID = Intrinsic::x86_avx512_mask_gather3siv4_si;
      break;
    case X86::BI__builtin_ia32_gather3siv8sf:
      IID = Intrinsic::x86_avx512_mask_gather3siv8_sf;
      break;
    case X86::BI__builtin_ia32_gather3siv8si:
      IID = Intrinsic::x86_avx512_mask_gather3siv8_si;
      break;
    case X86::BI__builtin_ia32_gathersiv8df:
      IID = Intrinsic::x86_avx512_mask_gather_dpd_512;
      break;
    case X86::BI__builtin_ia32_gathersiv16sf:
      IID = Intrinsic::x86_avx512_mask_gather_dps_512;
      break;
    case X86::BI__builtin_ia32_gatherdiv8df:
      IID = Intrinsic::x86_avx512_mask_gather_qpd_512;
      break;
    case X86::BI__builtin_ia32_gatherdiv16sf:
      IID = Intrinsic::x86_avx512_mask_gather_qps_512;
      break;
    case X86::BI__builtin_ia32_gathersiv8di:
      IID = Intrinsic::x86_avx512_mask_gather_dpq_512;
      break;
    case X86::BI__builtin_ia32_gathersiv16si:
      IID = Intrinsic::x86_avx512_mask_gather_dpi_512;
      break;
    case X86::BI__builtin_ia32_gatherdiv8di:
      IID = Intrinsic::x86_avx512_mask_gather_qpq_512;
      break;
    case X86::BI__builtin_ia32_gatherdiv16si:
      IID = Intrinsic::x86_avx512_mask_gather_qpi_512;
      break;
    }

    unsigned MinElts = std::min(
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements(),
        cast<llvm::FixedVectorType>(Ops[2]->getType())->getNumElements());
    Ops[3] = getMaskVecValue(*this, Ops[3], MinElts);
    Function *Intr = CGM.getIntrinsic(IID);
    return Builder.CreateCall(Intr, Ops);
  }

  case X86::BI__builtin_ia32_scattersiv8df:
  case X86::BI__builtin_ia32_scattersiv16sf:
  case X86::BI__builtin_ia32_scatterdiv8df:
  case X86::BI__builtin_ia32_scatterdiv16sf:
  case X86::BI__builtin_ia32_scattersiv8di:
  case X86::BI__builtin_ia32_scattersiv16si:
  case X86::BI__builtin_ia32_scatterdiv8di:
  case X86::BI__builtin_ia32_scatterdiv16si:
  case X86::BI__builtin_ia32_scatterdiv2df:
  case X86::BI__builtin_ia32_scatterdiv2di:
  case X86::BI__builtin_ia32_scatterdiv4df:
  case X86::BI__builtin_ia32_scatterdiv4di:
  case X86::BI__builtin_ia32_scatterdiv4sf:
  case X86::BI__builtin_ia32_scatterdiv4si:
  case X86::BI__builtin_ia32_scatterdiv8sf:
  case X86::BI__builtin_ia32_scatterdiv8si:
  case X86::BI__builtin_ia32_scattersiv2df:
  case X86::BI__builtin_ia32_scattersiv2di:
  case X86::BI__builtin_ia32_scattersiv4df:
  case X86::BI__builtin_ia32_scattersiv4di:
  case X86::BI__builtin_ia32_scattersiv4sf:
  case X86::BI__builtin_ia32_scattersiv4si:
  case X86::BI__builtin_ia32_scattersiv8sf:
  case X86::BI__builtin_ia32_scattersiv8si: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_scattersiv8df:
      IID = Intrinsic::x86_avx512_mask_scatter_dpd_512;
      break;
    case X86::BI__builtin_ia32_scattersiv16sf:
      IID = Intrinsic::x86_avx512_mask_scatter_dps_512;
      break;
    case X86::BI__builtin_ia32_scatterdiv8df:
      IID = Intrinsic::x86_avx512_mask_scatter_qpd_512;
      break;
    case X86::BI__builtin_ia32_scatterdiv16sf:
      IID = Intrinsic::x86_avx512_mask_scatter_qps_512;
      break;
    case X86::BI__builtin_ia32_scattersiv8di:
      IID = Intrinsic::x86_avx512_mask_scatter_dpq_512;
      break;
    case X86::BI__builtin_ia32_scattersiv16si:
      IID = Intrinsic::x86_avx512_mask_scatter_dpi_512;
      break;
    case X86::BI__builtin_ia32_scatterdiv8di:
      IID = Intrinsic::x86_avx512_mask_scatter_qpq_512;
      break;
    case X86::BI__builtin_ia32_scatterdiv16si:
      IID = Intrinsic::x86_avx512_mask_scatter_qpi_512;
      break;
    case X86::BI__builtin_ia32_scatterdiv2df:
      IID = Intrinsic::x86_avx512_mask_scatterdiv2_df;
      break;
    case X86::BI__builtin_ia32_scatterdiv2di:
      IID = Intrinsic::x86_avx512_mask_scatterdiv2_di;
      break;
    case X86::BI__builtin_ia32_scatterdiv4df:
      IID = Intrinsic::x86_avx512_mask_scatterdiv4_df;
      break;
    case X86::BI__builtin_ia32_scatterdiv4di:
      IID = Intrinsic::x86_avx512_mask_scatterdiv4_di;
      break;
    case X86::BI__builtin_ia32_scatterdiv4sf:
      IID = Intrinsic::x86_avx512_mask_scatterdiv4_sf;
      break;
    case X86::BI__builtin_ia32_scatterdiv4si:
      IID = Intrinsic::x86_avx512_mask_scatterdiv4_si;
      break;
    case X86::BI__builtin_ia32_scatterdiv8sf:
      IID = Intrinsic::x86_avx512_mask_scatterdiv8_sf;
      break;
    case X86::BI__builtin_ia32_scatterdiv8si:
      IID = Intrinsic::x86_avx512_mask_scatterdiv8_si;
      break;
    case X86::BI__builtin_ia32_scattersiv2df:
      IID = Intrinsic::x86_avx512_mask_scattersiv2_df;
      break;
    case X86::BI__builtin_ia32_scattersiv2di:
      IID = Intrinsic::x86_avx512_mask_scattersiv2_di;
      break;
    case X86::BI__builtin_ia32_scattersiv4df:
      IID = Intrinsic::x86_avx512_mask_scattersiv4_df;
      break;
    case X86::BI__builtin_ia32_scattersiv4di:
      IID = Intrinsic::x86_avx512_mask_scattersiv4_di;
      break;
    case X86::BI__builtin_ia32_scattersiv4sf:
      IID = Intrinsic::x86_avx512_mask_scattersiv4_sf;
      break;
    case X86::BI__builtin_ia32_scattersiv4si:
      IID = Intrinsic::x86_avx512_mask_scattersiv4_si;
      break;
    case X86::BI__builtin_ia32_scattersiv8sf:
      IID = Intrinsic::x86_avx512_mask_scattersiv8_sf;
      break;
    case X86::BI__builtin_ia32_scattersiv8si:
      IID = Intrinsic::x86_avx512_mask_scattersiv8_si;
      break;
    }

    unsigned MinElts = std::min(
        cast<llvm::FixedVectorType>(Ops[2]->getType())->getNumElements(),
        cast<llvm::FixedVectorType>(Ops[3]->getType())->getNumElements());
    Ops[1] = getMaskVecValue(*this, Ops[1], MinElts);
    Function *Intr = CGM.getIntrinsic(IID);
    return Builder.CreateCall(Intr, Ops);
  }

  case X86::BI__builtin_ia32_vextractf128_pd256:
  case X86::BI__builtin_ia32_vextractf128_ps256:
  case X86::BI__builtin_ia32_vextractf128_si256:
  case X86::BI__builtin_ia32_extract128i256:
  case X86::BI__builtin_ia32_extractf64x4_mask:
  case X86::BI__builtin_ia32_extractf32x4_mask:
  case X86::BI__builtin_ia32_extracti64x4_mask:
  case X86::BI__builtin_ia32_extracti32x4_mask:
  case X86::BI__builtin_ia32_extractf32x8_mask:
  case X86::BI__builtin_ia32_extracti32x8_mask:
  case X86::BI__builtin_ia32_extractf32x4_256_mask:
  case X86::BI__builtin_ia32_extracti32x4_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_256_mask:
  case X86::BI__builtin_ia32_extracti64x2_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_512_mask:
  case X86::BI__builtin_ia32_extracti64x2_512_mask: {
    auto *DstTy = cast<llvm::FixedVectorType>(ConvertType(E->getType()));
    unsigned NumElts = DstTy->getNumElements();
    unsigned SrcNumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    unsigned SubVectors = SrcNumElts / NumElts;
    unsigned Index = cast<ConstantInt>(Ops[1])->getZExtValue();
    assert(llvm::isPowerOf2_32(SubVectors) && "Expected power of 2 subvectors");
    Index &= SubVectors - 1; // Remove any extra bits.
    Index *= NumElts;

    int Indices[16];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i + Index;

    Value *Res = Builder.CreateShuffleVector(Ops[0], ArrayRef(Indices, NumElts),
                                             "extract");

    if (Ops.size() == 4)
      Res = EmitX86Select(*this, Ops[3], Res, Ops[2]);

    return Res;
  }
  case X86::BI__builtin_ia32_vinsertf128_pd256:
  case X86::BI__builtin_ia32_vinsertf128_ps256:
  case X86::BI__builtin_ia32_vinsertf128_si256:
  case X86::BI__builtin_ia32_insert128i256:
  case X86::BI__builtin_ia32_insertf64x4:
  case X86::BI__builtin_ia32_insertf32x4:
  case X86::BI__builtin_ia32_inserti64x4:
  case X86::BI__builtin_ia32_inserti32x4:
  case X86::BI__builtin_ia32_insertf32x8:
  case X86::BI__builtin_ia32_inserti32x8:
  case X86::BI__builtin_ia32_insertf32x4_256:
  case X86::BI__builtin_ia32_inserti32x4_256:
  case X86::BI__builtin_ia32_insertf64x2_256:
  case X86::BI__builtin_ia32_inserti64x2_256:
  case X86::BI__builtin_ia32_insertf64x2_512:
  case X86::BI__builtin_ia32_inserti64x2_512: {
    unsigned DstNumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    unsigned SrcNumElts =
        cast<llvm::FixedVectorType>(Ops[1]->getType())->getNumElements();
    unsigned SubVectors = DstNumElts / SrcNumElts;
    unsigned Index = cast<ConstantInt>(Ops[2])->getZExtValue();
    assert(llvm::isPowerOf2_32(SubVectors) && "Expected power of 2 subvectors");
    Index &= SubVectors - 1; // Remove any extra bits.
    Index *= SrcNumElts;

    int Indices[16];
    for (unsigned i = 0; i != DstNumElts; ++i)
      Indices[i] = (i >= SrcNumElts) ? SrcNumElts + (i % SrcNumElts) : i;

    Value *Op1 = Builder.CreateShuffleVector(
        Ops[1], ArrayRef(Indices, DstNumElts), "widen");

    for (unsigned i = 0; i != DstNumElts; ++i) {
      if (i >= Index && i < (Index + SrcNumElts))
        Indices[i] = (i - Index) + DstNumElts;
      else
        Indices[i] = i;
    }

    return Builder.CreateShuffleVector(Ops[0], Op1,
                                       ArrayRef(Indices, DstNumElts), "insert");
  }
  case X86::BI__builtin_ia32_pmovqd512_mask:
  case X86::BI__builtin_ia32_pmovwb512_mask: {
    Value *Res = Builder.CreateTrunc(Ops[0], Ops[1]->getType());
    return EmitX86Select(*this, Ops[2], Res, Ops[1]);
  }
  case X86::BI__builtin_ia32_pmovdb512_mask:
  case X86::BI__builtin_ia32_pmovdw512_mask:
  case X86::BI__builtin_ia32_pmovqw512_mask: {
    if (const auto *C = dyn_cast<Constant>(Ops[2]))
      if (C->isAllOnesValue())
        return Builder.CreateTrunc(Ops[0], Ops[1]->getType());

    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_pmovdb512_mask:
      IID = Intrinsic::x86_avx512_mask_pmov_db_512;
      break;
    case X86::BI__builtin_ia32_pmovdw512_mask:
      IID = Intrinsic::x86_avx512_mask_pmov_dw_512;
      break;
    case X86::BI__builtin_ia32_pmovqw512_mask:
      IID = Intrinsic::x86_avx512_mask_pmov_qw_512;
      break;
    }

    Function *Intr = CGM.getIntrinsic(IID);
    return Builder.CreateCall(Intr, Ops);
  }
  case X86::BI__builtin_ia32_pblendw128:
  case X86::BI__builtin_ia32_blendpd:
  case X86::BI__builtin_ia32_blendps:
  case X86::BI__builtin_ia32_blendpd256:
  case X86::BI__builtin_ia32_blendps256:
  case X86::BI__builtin_ia32_pblendw256:
  case X86::BI__builtin_ia32_pblendd128:
  case X86::BI__builtin_ia32_pblendd256: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    unsigned Imm = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    int Indices[16];
    // If there are more than 8 elements, the immediate is used twice so make
    // sure we handle that.
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = ((Imm >> (i % 8)) & 0x1) ? NumElts + i : i;

    return Builder.CreateShuffleVector(Ops[0], Ops[1],
                                       ArrayRef(Indices, NumElts), "blend");
  }
  case X86::BI__builtin_ia32_pshuflw:
  case X86::BI__builtin_ia32_pshuflw256:
  case X86::BI__builtin_ia32_pshuflw512: {
    uint32_t Imm = cast<llvm::ConstantInt>(Ops[1])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();

    // Splat the 8-bits of immediate 4 times to help the loop wrap around.
    Imm = (Imm & 0xff) * 0x01010101;

    int Indices[32];
    for (unsigned l = 0; l != NumElts; l += 8) {
      for (unsigned i = 0; i != 4; ++i) {
        Indices[l + i] = l + (Imm & 3);
        Imm >>= 2;
      }
      for (unsigned i = 4; i != 8; ++i)
        Indices[l + i] = l + i;
    }

    return Builder.CreateShuffleVector(Ops[0], ArrayRef(Indices, NumElts),
                                       "pshuflw");
  }
  case X86::BI__builtin_ia32_pshufhw:
  case X86::BI__builtin_ia32_pshufhw256:
  case X86::BI__builtin_ia32_pshufhw512: {
    uint32_t Imm = cast<llvm::ConstantInt>(Ops[1])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();

    // Splat the 8-bits of immediate 4 times to help the loop wrap around.
    Imm = (Imm & 0xff) * 0x01010101;

    int Indices[32];
    for (unsigned l = 0; l != NumElts; l += 8) {
      for (unsigned i = 0; i != 4; ++i)
        Indices[l + i] = l + i;
      for (unsigned i = 4; i != 8; ++i) {
        Indices[l + i] = l + 4 + (Imm & 3);
        Imm >>= 2;
      }
    }

    return Builder.CreateShuffleVector(Ops[0], ArrayRef(Indices, NumElts),
                                       "pshufhw");
  }
  case X86::BI__builtin_ia32_pshufd:
  case X86::BI__builtin_ia32_pshufd256:
  case X86::BI__builtin_ia32_pshufd512:
  case X86::BI__builtin_ia32_vpermilpd:
  case X86::BI__builtin_ia32_vpermilps:
  case X86::BI__builtin_ia32_vpermilpd256:
  case X86::BI__builtin_ia32_vpermilps256:
  case X86::BI__builtin_ia32_vpermilpd512:
  case X86::BI__builtin_ia32_vpermilps512: {
    uint32_t Imm = cast<llvm::ConstantInt>(Ops[1])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();
    unsigned NumLanes = Ty->getPrimitiveSizeInBits() / 128;
    unsigned NumLaneElts = NumElts / NumLanes;

    // Splat the 8-bits of immediate 4 times to help the loop wrap around.
    Imm = (Imm & 0xff) * 0x01010101;

    int Indices[16];
    for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
      for (unsigned i = 0; i != NumLaneElts; ++i) {
        Indices[i + l] = (Imm % NumLaneElts) + l;
        Imm /= NumLaneElts;
      }
    }

    return Builder.CreateShuffleVector(Ops[0], ArrayRef(Indices, NumElts),
                                       "permil");
  }
  case X86::BI__builtin_ia32_shufpd:
  case X86::BI__builtin_ia32_shufpd256:
  case X86::BI__builtin_ia32_shufpd512:
  case X86::BI__builtin_ia32_shufps:
  case X86::BI__builtin_ia32_shufps256:
  case X86::BI__builtin_ia32_shufps512: {
    uint32_t Imm = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();
    unsigned NumLanes = Ty->getPrimitiveSizeInBits() / 128;
    unsigned NumLaneElts = NumElts / NumLanes;

    // Splat the 8-bits of immediate 4 times to help the loop wrap around.
    Imm = (Imm & 0xff) * 0x01010101;

    int Indices[16];
    for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
      for (unsigned i = 0; i != NumLaneElts; ++i) {
        unsigned Index = Imm % NumLaneElts;
        Imm /= NumLaneElts;
        if (i >= (NumLaneElts / 2))
          Index += NumElts;
        Indices[l + i] = l + Index;
      }
    }

    return Builder.CreateShuffleVector(Ops[0], Ops[1],
                                       ArrayRef(Indices, NumElts), "shufp");
  }
  case X86::BI__builtin_ia32_permdi256:
  case X86::BI__builtin_ia32_permdf256:
  case X86::BI__builtin_ia32_permdi512:
  case X86::BI__builtin_ia32_permdf512: {
    unsigned Imm = cast<llvm::ConstantInt>(Ops[1])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();

    // These intrinsics operate on 256-bit lanes of four 64-bit elements.
    int Indices[8];
    for (unsigned l = 0; l != NumElts; l += 4)
      for (unsigned i = 0; i != 4; ++i)
        Indices[l + i] = l + ((Imm >> (2 * i)) & 0x3);

    return Builder.CreateShuffleVector(Ops[0], ArrayRef(Indices, NumElts),
                                       "perm");
  }
  case X86::BI__builtin_ia32_palignr128:
  case X86::BI__builtin_ia32_palignr256:
  case X86::BI__builtin_ia32_palignr512: {
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0xff;

    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    assert(NumElts % 16 == 0);

    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if (ShiftVal >= 32)
      return llvm::Constant::getNullValue(ConvertType(E->getType()));

    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    if (ShiftVal > 16) {
      ShiftVal -= 16;
      Ops[1] = Ops[0];
      Ops[0] = llvm::Constant::getNullValue(Ops[0]->getType());
    }

    int Indices[64];
    // 256-bit palignr operates on 128-bit lanes so we need to handle that
    for (unsigned l = 0; l != NumElts; l += 16) {
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = ShiftVal + i;
        if (Idx >= 16)
          Idx += NumElts - 16; // End of lane, switch operand.
        Indices[l + i] = Idx + l;
      }
    }

    return Builder.CreateShuffleVector(Ops[1], Ops[0],
                                       ArrayRef(Indices, NumElts), "palignr");
  }
  case X86::BI__builtin_ia32_alignd128:
  case X86::BI__builtin_ia32_alignd256:
  case X86::BI__builtin_ia32_alignd512:
  case X86::BI__builtin_ia32_alignq128:
  case X86::BI__builtin_ia32_alignq256:
  case X86::BI__builtin_ia32_alignq512: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0xff;

    // Mask the shift amount to width of a vector.
    ShiftVal &= NumElts - 1;

    int Indices[16];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i + ShiftVal;

    return Builder.CreateShuffleVector(Ops[1], Ops[0],
                                       ArrayRef(Indices, NumElts), "valign");
  }
  case X86::BI__builtin_ia32_shuf_f32x4_256:
  case X86::BI__builtin_ia32_shuf_f64x2_256:
  case X86::BI__builtin_ia32_shuf_i32x4_256:
  case X86::BI__builtin_ia32_shuf_i64x2_256:
  case X86::BI__builtin_ia32_shuf_f32x4:
  case X86::BI__builtin_ia32_shuf_f64x2:
  case X86::BI__builtin_ia32_shuf_i32x4:
  case X86::BI__builtin_ia32_shuf_i64x2: {
    unsigned Imm = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();
    auto *Ty = cast<llvm::FixedVectorType>(Ops[0]->getType());
    unsigned NumElts = Ty->getNumElements();
    unsigned NumLanes = Ty->getPrimitiveSizeInBits() == 512 ? 4 : 2;
    unsigned NumLaneElts = NumElts / NumLanes;

    int Indices[16];
    for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
      unsigned Index = (Imm % NumLanes) * NumLaneElts;
      Imm /= NumLanes; // Discard the bits we just used.
      if (l >= (NumElts / 2))
        Index += NumElts; // Switch to other source.
      for (unsigned i = 0; i != NumLaneElts; ++i) {
        Indices[l + i] = Index + i;
      }
    }

    return Builder.CreateShuffleVector(Ops[0], Ops[1],
                                       ArrayRef(Indices, NumElts), "shuf");
  }

  case X86::BI__builtin_ia32_vperm2f128_pd256:
  case X86::BI__builtin_ia32_vperm2f128_ps256:
  case X86::BI__builtin_ia32_vperm2f128_si256:
  case X86::BI__builtin_ia32_permti256: {
    unsigned Imm = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();

    // This takes a very simple approach since there are two lanes and a
    // shuffle can have 2 inputs. So we reserve the first input for the first
    // lane and the second input for the second lane. This may result in
    // duplicate sources, but this can be dealt with in the backend.

    Value *OutOps[2];
    int Indices[8];
    for (unsigned l = 0; l != 2; ++l) {
      // Determine the source for this lane.
      if (Imm & (1 << ((l * 4) + 3)))
        OutOps[l] = llvm::ConstantAggregateZero::get(Ops[0]->getType());
      else if (Imm & (1 << ((l * 4) + 1)))
        OutOps[l] = Ops[1];
      else
        OutOps[l] = Ops[0];

      for (unsigned i = 0; i != NumElts/2; ++i) {
        // Start with ith element of the source for this lane.
        unsigned Idx = (l * NumElts) + i;
        // If bit 0 of the immediate half is set, switch to the high half of
        // the source.
        if (Imm & (1 << (l * 4)))
          Idx += NumElts/2;
        Indices[(l * (NumElts/2)) + i] = Idx;
      }
    }

    return Builder.CreateShuffleVector(OutOps[0], OutOps[1],
                                       ArrayRef(Indices, NumElts), "vperm");
  }

  case X86::BI__builtin_ia32_pslldqi128_byteshift:
  case X86::BI__builtin_ia32_pslldqi256_byteshift:
  case X86::BI__builtin_ia32_pslldqi512_byteshift: {
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[1])->getZExtValue() & 0xff;
    auto *ResultType = cast<llvm::FixedVectorType>(Ops[0]->getType());
    // Builtin type is vXi64 so multiply by 8 to get bytes.
    unsigned NumElts = ResultType->getNumElements() * 8;

    // If pslldq is shifting the vector more than 15 bytes, emit zero.
    if (ShiftVal >= 16)
      return llvm::Constant::getNullValue(ResultType);

    int Indices[64];
    // 256/512-bit pslldq operates on 128-bit lanes so we need to handle that
    for (unsigned l = 0; l != NumElts; l += 16) {
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = NumElts + i - ShiftVal;
        if (Idx < NumElts) Idx -= NumElts - 16; // end of lane, switch operand.
        Indices[l + i] = Idx + l;
      }
    }

    auto *VecTy = llvm::FixedVectorType::get(Int8Ty, NumElts);
    Value *Cast = Builder.CreateBitCast(Ops[0], VecTy, "cast");
    Value *Zero = llvm::Constant::getNullValue(VecTy);
    Value *SV = Builder.CreateShuffleVector(
        Zero, Cast, ArrayRef(Indices, NumElts), "pslldq");
    return Builder.CreateBitCast(SV, Ops[0]->getType(), "cast");
  }
  case X86::BI__builtin_ia32_psrldqi128_byteshift:
  case X86::BI__builtin_ia32_psrldqi256_byteshift:
  case X86::BI__builtin_ia32_psrldqi512_byteshift: {
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[1])->getZExtValue() & 0xff;
    auto *ResultType = cast<llvm::FixedVectorType>(Ops[0]->getType());
    // Builtin type is vXi64 so multiply by 8 to get bytes.
    unsigned NumElts = ResultType->getNumElements() * 8;

    // If psrldq is shifting the vector more than 15 bytes, emit zero.
    if (ShiftVal >= 16)
      return llvm::Constant::getNullValue(ResultType);

    int Indices[64];
    // 256/512-bit psrldq operates on 128-bit lanes so we need to handle that
    for (unsigned l = 0; l != NumElts; l += 16) {
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = i + ShiftVal;
        if (Idx >= 16) Idx += NumElts - 16; // end of lane, switch operand.
        Indices[l + i] = Idx + l;
      }
    }

    auto *VecTy = llvm::FixedVectorType::get(Int8Ty, NumElts);
    Value *Cast = Builder.CreateBitCast(Ops[0], VecTy, "cast");
    Value *Zero = llvm::Constant::getNullValue(VecTy);
    Value *SV = Builder.CreateShuffleVector(
        Cast, Zero, ArrayRef(Indices, NumElts), "psrldq");
    return Builder.CreateBitCast(SV, ResultType, "cast");
  }
  case X86::BI__builtin_ia32_kshiftliqi:
  case X86::BI__builtin_ia32_kshiftlihi:
  case X86::BI__builtin_ia32_kshiftlisi:
  case X86::BI__builtin_ia32_kshiftlidi: {
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[1])->getZExtValue() & 0xff;
    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();

    if (ShiftVal >= NumElts)
      return llvm::Constant::getNullValue(Ops[0]->getType());

    Value *In = getMaskVecValue(*this, Ops[0], NumElts);

    int Indices[64];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = NumElts + i - ShiftVal;

    Value *Zero = llvm::Constant::getNullValue(In->getType());
    Value *SV = Builder.CreateShuffleVector(
        Zero, In, ArrayRef(Indices, NumElts), "kshiftl");
    return Builder.CreateBitCast(SV, Ops[0]->getType());
  }
  case X86::BI__builtin_ia32_kshiftriqi:
  case X86::BI__builtin_ia32_kshiftrihi:
  case X86::BI__builtin_ia32_kshiftrisi:
  case X86::BI__builtin_ia32_kshiftridi: {
    unsigned ShiftVal = cast<llvm::ConstantInt>(Ops[1])->getZExtValue() & 0xff;
    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();

    if (ShiftVal >= NumElts)
      return llvm::Constant::getNullValue(Ops[0]->getType());

    Value *In = getMaskVecValue(*this, Ops[0], NumElts);

    int Indices[64];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i + ShiftVal;

    Value *Zero = llvm::Constant::getNullValue(In->getType());
    Value *SV = Builder.CreateShuffleVector(
        In, Zero, ArrayRef(Indices, NumElts), "kshiftr");
    return Builder.CreateBitCast(SV, Ops[0]->getType());
  }
  case X86::BI__builtin_ia32_movnti:
  case X86::BI__builtin_ia32_movnti64:
  case X86::BI__builtin_ia32_movntsd:
  case X86::BI__builtin_ia32_movntss: {
    llvm::MDNode *Node = llvm::MDNode::get(
        getLLVMContext(), llvm::ConstantAsMetadata::get(Builder.getInt32(1)));

    Value *Ptr = Ops[0];
    Value *Src = Ops[1];

    // Extract the 0'th element of the source vector.
    if (BuiltinID == X86::BI__builtin_ia32_movntsd ||
        BuiltinID == X86::BI__builtin_ia32_movntss)
      Src = Builder.CreateExtractElement(Src, (uint64_t)0, "extract");

    // Unaligned nontemporal store of the scalar value.
    StoreInst *SI = Builder.CreateDefaultAlignedStore(Src, Ptr);
    SI->setMetadata(llvm::LLVMContext::MD_nontemporal, Node);
    SI->setAlignment(llvm::Align(1));
    return SI;
  }
  // Rotate is a special case of funnel shift - 1st 2 args are the same.
  case X86::BI__builtin_ia32_vprotb:
  case X86::BI__builtin_ia32_vprotw:
  case X86::BI__builtin_ia32_vprotd:
  case X86::BI__builtin_ia32_vprotq:
  case X86::BI__builtin_ia32_vprotbi:
  case X86::BI__builtin_ia32_vprotwi:
  case X86::BI__builtin_ia32_vprotdi:
  case X86::BI__builtin_ia32_vprotqi:
  case X86::BI__builtin_ia32_prold128:
  case X86::BI__builtin_ia32_prold256:
  case X86::BI__builtin_ia32_prold512:
  case X86::BI__builtin_ia32_prolq128:
  case X86::BI__builtin_ia32_prolq256:
  case X86::BI__builtin_ia32_prolq512:
  case X86::BI__builtin_ia32_prolvd128:
  case X86::BI__builtin_ia32_prolvd256:
  case X86::BI__builtin_ia32_prolvd512:
  case X86::BI__builtin_ia32_prolvq128:
  case X86::BI__builtin_ia32_prolvq256:
  case X86::BI__builtin_ia32_prolvq512:
    return EmitX86FunnelShift(*this, Ops[0], Ops[0], Ops[1], false);
  case X86::BI__builtin_ia32_prord128:
  case X86::BI__builtin_ia32_prord256:
  case X86::BI__builtin_ia32_prord512:
  case X86::BI__builtin_ia32_prorq128:
  case X86::BI__builtin_ia32_prorq256:
  case X86::BI__builtin_ia32_prorq512:
  case X86::BI__builtin_ia32_prorvd128:
  case X86::BI__builtin_ia32_prorvd256:
  case X86::BI__builtin_ia32_prorvd512:
  case X86::BI__builtin_ia32_prorvq128:
  case X86::BI__builtin_ia32_prorvq256:
  case X86::BI__builtin_ia32_prorvq512:
    return EmitX86FunnelShift(*this, Ops[0], Ops[0], Ops[1], true);
  case X86::BI__builtin_ia32_selectb_128:
  case X86::BI__builtin_ia32_selectb_256:
  case X86::BI__builtin_ia32_selectb_512:
  case X86::BI__builtin_ia32_selectw_128:
  case X86::BI__builtin_ia32_selectw_256:
  case X86::BI__builtin_ia32_selectw_512:
  case X86::BI__builtin_ia32_selectd_128:
  case X86::BI__builtin_ia32_selectd_256:
  case X86::BI__builtin_ia32_selectd_512:
  case X86::BI__builtin_ia32_selectq_128:
  case X86::BI__builtin_ia32_selectq_256:
  case X86::BI__builtin_ia32_selectq_512:
  case X86::BI__builtin_ia32_selectph_128:
  case X86::BI__builtin_ia32_selectph_256:
  case X86::BI__builtin_ia32_selectph_512:
  case X86::BI__builtin_ia32_selectpbf_128:
  case X86::BI__builtin_ia32_selectpbf_256:
  case X86::BI__builtin_ia32_selectpbf_512:
  case X86::BI__builtin_ia32_selectps_128:
  case X86::BI__builtin_ia32_selectps_256:
  case X86::BI__builtin_ia32_selectps_512:
  case X86::BI__builtin_ia32_selectpd_128:
  case X86::BI__builtin_ia32_selectpd_256:
  case X86::BI__builtin_ia32_selectpd_512:
    return EmitX86Select(*this, Ops[0], Ops[1], Ops[2]);
  case X86::BI__builtin_ia32_selectsh_128:
  case X86::BI__builtin_ia32_selectsbf_128:
  case X86::BI__builtin_ia32_selectss_128:
  case X86::BI__builtin_ia32_selectsd_128: {
    Value *A = Builder.CreateExtractElement(Ops[1], (uint64_t)0);
    Value *B = Builder.CreateExtractElement(Ops[2], (uint64_t)0);
    A = EmitX86ScalarSelect(*this, Ops[0], A, B);
    return Builder.CreateInsertElement(Ops[1], A, (uint64_t)0);
  }
  case X86::BI__builtin_ia32_cmpb128_mask:
  case X86::BI__builtin_ia32_cmpb256_mask:
  case X86::BI__builtin_ia32_cmpb512_mask:
  case X86::BI__builtin_ia32_cmpw128_mask:
  case X86::BI__builtin_ia32_cmpw256_mask:
  case X86::BI__builtin_ia32_cmpw512_mask:
  case X86::BI__builtin_ia32_cmpd128_mask:
  case X86::BI__builtin_ia32_cmpd256_mask:
  case X86::BI__builtin_ia32_cmpd512_mask:
  case X86::BI__builtin_ia32_cmpq128_mask:
  case X86::BI__builtin_ia32_cmpq256_mask:
  case X86::BI__builtin_ia32_cmpq512_mask: {
    unsigned CC = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0x7;
    return EmitX86MaskedCompare(*this, CC, true, Ops);
  }
  case X86::BI__builtin_ia32_ucmpb128_mask:
  case X86::BI__builtin_ia32_ucmpb256_mask:
  case X86::BI__builtin_ia32_ucmpb512_mask:
  case X86::BI__builtin_ia32_ucmpw128_mask:
  case X86::BI__builtin_ia32_ucmpw256_mask:
  case X86::BI__builtin_ia32_ucmpw512_mask:
  case X86::BI__builtin_ia32_ucmpd128_mask:
  case X86::BI__builtin_ia32_ucmpd256_mask:
  case X86::BI__builtin_ia32_ucmpd512_mask:
  case X86::BI__builtin_ia32_ucmpq128_mask:
  case X86::BI__builtin_ia32_ucmpq256_mask:
  case X86::BI__builtin_ia32_ucmpq512_mask: {
    unsigned CC = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0x7;
    return EmitX86MaskedCompare(*this, CC, false, Ops);
  }
  case X86::BI__builtin_ia32_vpcomb:
  case X86::BI__builtin_ia32_vpcomw:
  case X86::BI__builtin_ia32_vpcomd:
  case X86::BI__builtin_ia32_vpcomq:
    return EmitX86vpcom(*this, Ops, true);
  case X86::BI__builtin_ia32_vpcomub:
  case X86::BI__builtin_ia32_vpcomuw:
  case X86::BI__builtin_ia32_vpcomud:
  case X86::BI__builtin_ia32_vpcomuq:
    return EmitX86vpcom(*this, Ops, false);

  case X86::BI__builtin_ia32_kortestcqi:
  case X86::BI__builtin_ia32_kortestchi:
  case X86::BI__builtin_ia32_kortestcsi:
  case X86::BI__builtin_ia32_kortestcdi: {
    Value *Or = EmitX86MaskLogic(*this, Instruction::Or, Ops);
    Value *C = llvm::Constant::getAllOnesValue(Ops[0]->getType());
    Value *Cmp = Builder.CreateICmpEQ(Or, C);
    return Builder.CreateZExt(Cmp, ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_kortestzqi:
  case X86::BI__builtin_ia32_kortestzhi:
  case X86::BI__builtin_ia32_kortestzsi:
  case X86::BI__builtin_ia32_kortestzdi: {
    Value *Or = EmitX86MaskLogic(*this, Instruction::Or, Ops);
    Value *C = llvm::Constant::getNullValue(Ops[0]->getType());
    Value *Cmp = Builder.CreateICmpEQ(Or, C);
    return Builder.CreateZExt(Cmp, ConvertType(E->getType()));
  }

  case X86::BI__builtin_ia32_ktestcqi:
  case X86::BI__builtin_ia32_ktestzqi:
  case X86::BI__builtin_ia32_ktestchi:
  case X86::BI__builtin_ia32_ktestzhi:
  case X86::BI__builtin_ia32_ktestcsi:
  case X86::BI__builtin_ia32_ktestzsi:
  case X86::BI__builtin_ia32_ktestcdi:
  case X86::BI__builtin_ia32_ktestzdi: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_ktestcqi:
      IID = Intrinsic::x86_avx512_ktestc_b;
      break;
    case X86::BI__builtin_ia32_ktestzqi:
      IID = Intrinsic::x86_avx512_ktestz_b;
      break;
    case X86::BI__builtin_ia32_ktestchi:
      IID = Intrinsic::x86_avx512_ktestc_w;
      break;
    case X86::BI__builtin_ia32_ktestzhi:
      IID = Intrinsic::x86_avx512_ktestz_w;
      break;
    case X86::BI__builtin_ia32_ktestcsi:
      IID = Intrinsic::x86_avx512_ktestc_d;
      break;
    case X86::BI__builtin_ia32_ktestzsi:
      IID = Intrinsic::x86_avx512_ktestz_d;
      break;
    case X86::BI__builtin_ia32_ktestcdi:
      IID = Intrinsic::x86_avx512_ktestc_q;
      break;
    case X86::BI__builtin_ia32_ktestzdi:
      IID = Intrinsic::x86_avx512_ktestz_q;
      break;
    }

    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
    Value *LHS = getMaskVecValue(*this, Ops[0], NumElts);
    Value *RHS = getMaskVecValue(*this, Ops[1], NumElts);
    Function *Intr = CGM.getIntrinsic(IID);
    return Builder.CreateCall(Intr, {LHS, RHS});
  }

  case X86::BI__builtin_ia32_kaddqi:
  case X86::BI__builtin_ia32_kaddhi:
  case X86::BI__builtin_ia32_kaddsi:
  case X86::BI__builtin_ia32_kadddi: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_kaddqi:
      IID = Intrinsic::x86_avx512_kadd_b;
      break;
    case X86::BI__builtin_ia32_kaddhi:
      IID = Intrinsic::x86_avx512_kadd_w;
      break;
    case X86::BI__builtin_ia32_kaddsi:
      IID = Intrinsic::x86_avx512_kadd_d;
      break;
    case X86::BI__builtin_ia32_kadddi:
      IID = Intrinsic::x86_avx512_kadd_q;
      break;
    }

    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
    Value *LHS = getMaskVecValue(*this, Ops[0], NumElts);
    Value *RHS = getMaskVecValue(*this, Ops[1], NumElts);
    Function *Intr = CGM.getIntrinsic(IID);
    Value *Res = Builder.CreateCall(Intr, {LHS, RHS});
    return Builder.CreateBitCast(Res, Ops[0]->getType());
  }
  case X86::BI__builtin_ia32_kandqi:
  case X86::BI__builtin_ia32_kandhi:
  case X86::BI__builtin_ia32_kandsi:
  case X86::BI__builtin_ia32_kanddi:
    return EmitX86MaskLogic(*this, Instruction::And, Ops);
  case X86::BI__builtin_ia32_kandnqi:
  case X86::BI__builtin_ia32_kandnhi:
  case X86::BI__builtin_ia32_kandnsi:
  case X86::BI__builtin_ia32_kandndi:
    return EmitX86MaskLogic(*this, Instruction::And, Ops, true);
  case X86::BI__builtin_ia32_korqi:
  case X86::BI__builtin_ia32_korhi:
  case X86::BI__builtin_ia32_korsi:
  case X86::BI__builtin_ia32_kordi:
    return EmitX86MaskLogic(*this, Instruction::Or, Ops);
  case X86::BI__builtin_ia32_kxnorqi:
  case X86::BI__builtin_ia32_kxnorhi:
  case X86::BI__builtin_ia32_kxnorsi:
  case X86::BI__builtin_ia32_kxnordi:
    return EmitX86MaskLogic(*this, Instruction::Xor, Ops, true);
  case X86::BI__builtin_ia32_kxorqi:
  case X86::BI__builtin_ia32_kxorhi:
  case X86::BI__builtin_ia32_kxorsi:
  case X86::BI__builtin_ia32_kxordi:
    return EmitX86MaskLogic(*this, Instruction::Xor,  Ops);
  case X86::BI__builtin_ia32_knotqi:
  case X86::BI__builtin_ia32_knothi:
  case X86::BI__builtin_ia32_knotsi:
  case X86::BI__builtin_ia32_knotdi: {
    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
    Value *Res = getMaskVecValue(*this, Ops[0], NumElts);
    return Builder.CreateBitCast(Builder.CreateNot(Res),
                                 Ops[0]->getType());
  }
  case X86::BI__builtin_ia32_kmovb:
  case X86::BI__builtin_ia32_kmovw:
  case X86::BI__builtin_ia32_kmovd:
  case X86::BI__builtin_ia32_kmovq: {
    // Bitcast to vXi1 type and then back to integer. This gets the mask
    // register type into the IR, but might be optimized out depending on
    // what's around it.
    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
    Value *Res = getMaskVecValue(*this, Ops[0], NumElts);
    return Builder.CreateBitCast(Res, Ops[0]->getType());
  }

  case X86::BI__builtin_ia32_kunpckdi:
  case X86::BI__builtin_ia32_kunpcksi:
  case X86::BI__builtin_ia32_kunpckhi: {
    unsigned NumElts = Ops[0]->getType()->getIntegerBitWidth();
    Value *LHS = getMaskVecValue(*this, Ops[0], NumElts);
    Value *RHS = getMaskVecValue(*this, Ops[1], NumElts);
    int Indices[64];
    for (unsigned i = 0; i != NumElts; ++i)
      Indices[i] = i;

    // First extract half of each vector. This gives better codegen than
    // doing it in a single shuffle.
    LHS = Builder.CreateShuffleVector(LHS, LHS, ArrayRef(Indices, NumElts / 2));
    RHS = Builder.CreateShuffleVector(RHS, RHS, ArrayRef(Indices, NumElts / 2));
    // Concat the vectors.
    // NOTE: Operands are swapped to match the intrinsic definition.
    Value *Res =
        Builder.CreateShuffleVector(RHS, LHS, ArrayRef(Indices, NumElts));
    return Builder.CreateBitCast(Res, Ops[0]->getType());
  }

  case X86::BI__builtin_ia32_vplzcntd_128:
  case X86::BI__builtin_ia32_vplzcntd_256:
  case X86::BI__builtin_ia32_vplzcntd_512:
  case X86::BI__builtin_ia32_vplzcntq_128:
  case X86::BI__builtin_ia32_vplzcntq_256:
  case X86::BI__builtin_ia32_vplzcntq_512: {
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Ops[0]->getType());
    return Builder.CreateCall(F, {Ops[0],Builder.getInt1(false)});
  }
  case X86::BI__builtin_ia32_sqrtss:
  case X86::BI__builtin_ia32_sqrtsd: {
    Value *A = Builder.CreateExtractElement(Ops[0], (uint64_t)0);
    Function *F;
    if (Builder.getIsFPConstrained()) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      F = CGM.getIntrinsic(Intrinsic::experimental_constrained_sqrt,
                           A->getType());
      A = Builder.CreateConstrainedFPCall(F, {A});
    } else {
      F = CGM.getIntrinsic(Intrinsic::sqrt, A->getType());
      A = Builder.CreateCall(F, {A});
    }
    return Builder.CreateInsertElement(Ops[0], A, (uint64_t)0);
  }
  case X86::BI__builtin_ia32_sqrtsh_round_mask:
  case X86::BI__builtin_ia32_sqrtsd_round_mask:
  case X86::BI__builtin_ia32_sqrtss_round_mask: {
    unsigned CC = cast<llvm::ConstantInt>(Ops[4])->getZExtValue();
    // Support only if the rounding mode is 4 (AKA CUR_DIRECTION),
    // otherwise keep the intrinsic.
    if (CC != 4) {
      Intrinsic::ID IID;

      switch (BuiltinID) {
      default:
        llvm_unreachable("Unsupported intrinsic!");
      case X86::BI__builtin_ia32_sqrtsh_round_mask:
        IID = Intrinsic::x86_avx512fp16_mask_sqrt_sh;
        break;
      case X86::BI__builtin_ia32_sqrtsd_round_mask:
        IID = Intrinsic::x86_avx512_mask_sqrt_sd;
        break;
      case X86::BI__builtin_ia32_sqrtss_round_mask:
        IID = Intrinsic::x86_avx512_mask_sqrt_ss;
        break;
      }
      return Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
    }
    Value *A = Builder.CreateExtractElement(Ops[1], (uint64_t)0);
    Function *F;
    if (Builder.getIsFPConstrained()) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      F = CGM.getIntrinsic(Intrinsic::experimental_constrained_sqrt,
                           A->getType());
      A = Builder.CreateConstrainedFPCall(F, A);
    } else {
      F = CGM.getIntrinsic(Intrinsic::sqrt, A->getType());
      A = Builder.CreateCall(F, A);
    }
    Value *Src = Builder.CreateExtractElement(Ops[2], (uint64_t)0);
    A = EmitX86ScalarSelect(*this, Ops[3], A, Src);
    return Builder.CreateInsertElement(Ops[0], A, (uint64_t)0);
  }
  case X86::BI__builtin_ia32_sqrtpd256:
  case X86::BI__builtin_ia32_sqrtpd:
  case X86::BI__builtin_ia32_sqrtps256:
  case X86::BI__builtin_ia32_sqrtps:
  case X86::BI__builtin_ia32_sqrtph256:
  case X86::BI__builtin_ia32_sqrtph:
  case X86::BI__builtin_ia32_sqrtph512:
  case X86::BI__builtin_ia32_vsqrtbf16256:
  case X86::BI__builtin_ia32_vsqrtbf16:
  case X86::BI__builtin_ia32_vsqrtbf16512:
  case X86::BI__builtin_ia32_sqrtps512:
  case X86::BI__builtin_ia32_sqrtpd512: {
    if (Ops.size() == 2) {
      unsigned CC = cast<llvm::ConstantInt>(Ops[1])->getZExtValue();
      // Support only if the rounding mode is 4 (AKA CUR_DIRECTION),
      // otherwise keep the intrinsic.
      if (CC != 4) {
        Intrinsic::ID IID;

        switch (BuiltinID) {
        default:
          llvm_unreachable("Unsupported intrinsic!");
        case X86::BI__builtin_ia32_sqrtph512:
          IID = Intrinsic::x86_avx512fp16_sqrt_ph_512;
          break;
        case X86::BI__builtin_ia32_sqrtps512:
          IID = Intrinsic::x86_avx512_sqrt_ps_512;
          break;
        case X86::BI__builtin_ia32_sqrtpd512:
          IID = Intrinsic::x86_avx512_sqrt_pd_512;
          break;
        }
        return Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
      }
    }
    if (Builder.getIsFPConstrained()) {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_sqrt,
                                     Ops[0]->getType());
      return Builder.CreateConstrainedFPCall(F, Ops[0]);
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::sqrt, Ops[0]->getType());
      return Builder.CreateCall(F, Ops[0]);
    }
  }

  case X86::BI__builtin_ia32_pmuludq128:
  case X86::BI__builtin_ia32_pmuludq256:
  case X86::BI__builtin_ia32_pmuludq512:
    return EmitX86Muldq(*this, /*IsSigned*/false, Ops);

  case X86::BI__builtin_ia32_pmuldq128:
  case X86::BI__builtin_ia32_pmuldq256:
  case X86::BI__builtin_ia32_pmuldq512:
    return EmitX86Muldq(*this, /*IsSigned*/true, Ops);

  case X86::BI__builtin_ia32_pternlogd512_mask:
  case X86::BI__builtin_ia32_pternlogq512_mask:
  case X86::BI__builtin_ia32_pternlogd128_mask:
  case X86::BI__builtin_ia32_pternlogd256_mask:
  case X86::BI__builtin_ia32_pternlogq128_mask:
  case X86::BI__builtin_ia32_pternlogq256_mask:
    return EmitX86Ternlog(*this, /*ZeroMask*/false, Ops);

  case X86::BI__builtin_ia32_pternlogd512_maskz:
  case X86::BI__builtin_ia32_pternlogq512_maskz:
  case X86::BI__builtin_ia32_pternlogd128_maskz:
  case X86::BI__builtin_ia32_pternlogd256_maskz:
  case X86::BI__builtin_ia32_pternlogq128_maskz:
  case X86::BI__builtin_ia32_pternlogq256_maskz:
    return EmitX86Ternlog(*this, /*ZeroMask*/true, Ops);

  case X86::BI__builtin_ia32_vpshldd128:
  case X86::BI__builtin_ia32_vpshldd256:
  case X86::BI__builtin_ia32_vpshldd512:
  case X86::BI__builtin_ia32_vpshldq128:
  case X86::BI__builtin_ia32_vpshldq256:
  case X86::BI__builtin_ia32_vpshldq512:
  case X86::BI__builtin_ia32_vpshldw128:
  case X86::BI__builtin_ia32_vpshldw256:
  case X86::BI__builtin_ia32_vpshldw512:
    return EmitX86FunnelShift(*this, Ops[0], Ops[1], Ops[2], false);

  case X86::BI__builtin_ia32_vpshrdd128:
  case X86::BI__builtin_ia32_vpshrdd256:
  case X86::BI__builtin_ia32_vpshrdd512:
  case X86::BI__builtin_ia32_vpshrdq128:
  case X86::BI__builtin_ia32_vpshrdq256:
  case X86::BI__builtin_ia32_vpshrdq512:
  case X86::BI__builtin_ia32_vpshrdw128:
  case X86::BI__builtin_ia32_vpshrdw256:
  case X86::BI__builtin_ia32_vpshrdw512:
    // Ops 0 and 1 are swapped.
    return EmitX86FunnelShift(*this, Ops[1], Ops[0], Ops[2], true);

  case X86::BI__builtin_ia32_vpshldvd128:
  case X86::BI__builtin_ia32_vpshldvd256:
  case X86::BI__builtin_ia32_vpshldvd512:
  case X86::BI__builtin_ia32_vpshldvq128:
  case X86::BI__builtin_ia32_vpshldvq256:
  case X86::BI__builtin_ia32_vpshldvq512:
  case X86::BI__builtin_ia32_vpshldvw128:
  case X86::BI__builtin_ia32_vpshldvw256:
  case X86::BI__builtin_ia32_vpshldvw512:
    return EmitX86FunnelShift(*this, Ops[0], Ops[1], Ops[2], false);

  case X86::BI__builtin_ia32_vpshrdvd128:
  case X86::BI__builtin_ia32_vpshrdvd256:
  case X86::BI__builtin_ia32_vpshrdvd512:
  case X86::BI__builtin_ia32_vpshrdvq128:
  case X86::BI__builtin_ia32_vpshrdvq256:
  case X86::BI__builtin_ia32_vpshrdvq512:
  case X86::BI__builtin_ia32_vpshrdvw128:
  case X86::BI__builtin_ia32_vpshrdvw256:
  case X86::BI__builtin_ia32_vpshrdvw512:
    // Ops 0 and 1 are swapped.
    return EmitX86FunnelShift(*this, Ops[1], Ops[0], Ops[2], true);

  // Reductions
  case X86::BI__builtin_ia32_reduce_fadd_pd512:
  case X86::BI__builtin_ia32_reduce_fadd_ps512:
  case X86::BI__builtin_ia32_reduce_fadd_ph512:
  case X86::BI__builtin_ia32_reduce_fadd_ph256:
  case X86::BI__builtin_ia32_reduce_fadd_ph128: {
    Function *F =
        CGM.getIntrinsic(Intrinsic::vector_reduce_fadd, Ops[1]->getType());
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.getFastMathFlags().setAllowReassoc();
    return Builder.CreateCall(F, {Ops[0], Ops[1]});
  }
  case X86::BI__builtin_ia32_reduce_fmul_pd512:
  case X86::BI__builtin_ia32_reduce_fmul_ps512:
  case X86::BI__builtin_ia32_reduce_fmul_ph512:
  case X86::BI__builtin_ia32_reduce_fmul_ph256:
  case X86::BI__builtin_ia32_reduce_fmul_ph128: {
    Function *F =
        CGM.getIntrinsic(Intrinsic::vector_reduce_fmul, Ops[1]->getType());
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.getFastMathFlags().setAllowReassoc();
    return Builder.CreateCall(F, {Ops[0], Ops[1]});
  }
  case X86::BI__builtin_ia32_reduce_fmax_pd512:
  case X86::BI__builtin_ia32_reduce_fmax_ps512:
  case X86::BI__builtin_ia32_reduce_fmax_ph512:
  case X86::BI__builtin_ia32_reduce_fmax_ph256:
  case X86::BI__builtin_ia32_reduce_fmax_ph128: {
    Function *F =
        CGM.getIntrinsic(Intrinsic::vector_reduce_fmax, Ops[0]->getType());
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.getFastMathFlags().setNoNaNs();
    return Builder.CreateCall(F, {Ops[0]});
  }
  case X86::BI__builtin_ia32_reduce_fmin_pd512:
  case X86::BI__builtin_ia32_reduce_fmin_ps512:
  case X86::BI__builtin_ia32_reduce_fmin_ph512:
  case X86::BI__builtin_ia32_reduce_fmin_ph256:
  case X86::BI__builtin_ia32_reduce_fmin_ph128: {
    Function *F =
        CGM.getIntrinsic(Intrinsic::vector_reduce_fmin, Ops[0]->getType());
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.getFastMathFlags().setNoNaNs();
    return Builder.CreateCall(F, {Ops[0]});
  }

  case X86::BI__builtin_ia32_rdrand16_step:
  case X86::BI__builtin_ia32_rdrand32_step:
  case X86::BI__builtin_ia32_rdrand64_step:
  case X86::BI__builtin_ia32_rdseed16_step:
  case X86::BI__builtin_ia32_rdseed32_step:
  case X86::BI__builtin_ia32_rdseed64_step: {
    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_rdrand16_step:
      ID = Intrinsic::x86_rdrand_16;
      break;
    case X86::BI__builtin_ia32_rdrand32_step:
      ID = Intrinsic::x86_rdrand_32;
      break;
    case X86::BI__builtin_ia32_rdrand64_step:
      ID = Intrinsic::x86_rdrand_64;
      break;
    case X86::BI__builtin_ia32_rdseed16_step:
      ID = Intrinsic::x86_rdseed_16;
      break;
    case X86::BI__builtin_ia32_rdseed32_step:
      ID = Intrinsic::x86_rdseed_32;
      break;
    case X86::BI__builtin_ia32_rdseed64_step:
      ID = Intrinsic::x86_rdseed_64;
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(ID));
    Builder.CreateDefaultAlignedStore(Builder.CreateExtractValue(Call, 0),
                                      Ops[0]);
    return Builder.CreateExtractValue(Call, 1);
  }
  case X86::BI__builtin_ia32_addcarryx_u32:
  case X86::BI__builtin_ia32_addcarryx_u64:
  case X86::BI__builtin_ia32_subborrow_u32:
  case X86::BI__builtin_ia32_subborrow_u64: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_addcarryx_u32:
      IID = Intrinsic::x86_addcarry_32;
      break;
    case X86::BI__builtin_ia32_addcarryx_u64:
      IID = Intrinsic::x86_addcarry_64;
      break;
    case X86::BI__builtin_ia32_subborrow_u32:
      IID = Intrinsic::x86_subborrow_32;
      break;
    case X86::BI__builtin_ia32_subborrow_u64:
      IID = Intrinsic::x86_subborrow_64;
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID),
                                     { Ops[0], Ops[1], Ops[2] });
    Builder.CreateDefaultAlignedStore(Builder.CreateExtractValue(Call, 1),
                                      Ops[3]);
    return Builder.CreateExtractValue(Call, 0);
  }

  case X86::BI__builtin_ia32_fpclassps128_mask:
  case X86::BI__builtin_ia32_fpclassps256_mask:
  case X86::BI__builtin_ia32_fpclassps512_mask:
  case X86::BI__builtin_ia32_vfpclassbf16128_mask:
  case X86::BI__builtin_ia32_vfpclassbf16256_mask:
  case X86::BI__builtin_ia32_vfpclassbf16512_mask:
  case X86::BI__builtin_ia32_fpclassph128_mask:
  case X86::BI__builtin_ia32_fpclassph256_mask:
  case X86::BI__builtin_ia32_fpclassph512_mask:
  case X86::BI__builtin_ia32_fpclasspd128_mask:
  case X86::BI__builtin_ia32_fpclasspd256_mask:
  case X86::BI__builtin_ia32_fpclasspd512_mask: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    Value *MaskIn = Ops[2];
    Ops.erase(&Ops[2]);

    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_vfpclassbf16128_mask:
      ID = Intrinsic::x86_avx10_fpclass_bf16_128;
      break;
    case X86::BI__builtin_ia32_vfpclassbf16256_mask:
      ID = Intrinsic::x86_avx10_fpclass_bf16_256;
      break;
    case X86::BI__builtin_ia32_vfpclassbf16512_mask:
      ID = Intrinsic::x86_avx10_fpclass_bf16_512;
      break;
    case X86::BI__builtin_ia32_fpclassph128_mask:
      ID = Intrinsic::x86_avx512fp16_fpclass_ph_128;
      break;
    case X86::BI__builtin_ia32_fpclassph256_mask:
      ID = Intrinsic::x86_avx512fp16_fpclass_ph_256;
      break;
    case X86::BI__builtin_ia32_fpclassph512_mask:
      ID = Intrinsic::x86_avx512fp16_fpclass_ph_512;
      break;
    case X86::BI__builtin_ia32_fpclassps128_mask:
      ID = Intrinsic::x86_avx512_fpclass_ps_128;
      break;
    case X86::BI__builtin_ia32_fpclassps256_mask:
      ID = Intrinsic::x86_avx512_fpclass_ps_256;
      break;
    case X86::BI__builtin_ia32_fpclassps512_mask:
      ID = Intrinsic::x86_avx512_fpclass_ps_512;
      break;
    case X86::BI__builtin_ia32_fpclasspd128_mask:
      ID = Intrinsic::x86_avx512_fpclass_pd_128;
      break;
    case X86::BI__builtin_ia32_fpclasspd256_mask:
      ID = Intrinsic::x86_avx512_fpclass_pd_256;
      break;
    case X86::BI__builtin_ia32_fpclasspd512_mask:
      ID = Intrinsic::x86_avx512_fpclass_pd_512;
      break;
    }

    Value *Fpclass = Builder.CreateCall(CGM.getIntrinsic(ID), Ops);
    return EmitX86MaskedCompareResult(*this, Fpclass, NumElts, MaskIn);
  }

  case X86::BI__builtin_ia32_vp2intersect_q_512:
  case X86::BI__builtin_ia32_vp2intersect_q_256:
  case X86::BI__builtin_ia32_vp2intersect_q_128:
  case X86::BI__builtin_ia32_vp2intersect_d_512:
  case X86::BI__builtin_ia32_vp2intersect_d_256:
  case X86::BI__builtin_ia32_vp2intersect_d_128: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    Intrinsic::ID ID;

    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_vp2intersect_q_512:
      ID = Intrinsic::x86_avx512_vp2intersect_q_512;
      break;
    case X86::BI__builtin_ia32_vp2intersect_q_256:
      ID = Intrinsic::x86_avx512_vp2intersect_q_256;
      break;
    case X86::BI__builtin_ia32_vp2intersect_q_128:
      ID = Intrinsic::x86_avx512_vp2intersect_q_128;
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_512:
      ID = Intrinsic::x86_avx512_vp2intersect_d_512;
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_256:
      ID = Intrinsic::x86_avx512_vp2intersect_d_256;
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_128:
      ID = Intrinsic::x86_avx512_vp2intersect_d_128;
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(ID), {Ops[0], Ops[1]});
    Value *Result = Builder.CreateExtractValue(Call, 0);
    Result = EmitX86MaskedCompareResult(*this, Result, NumElts, nullptr);
    Builder.CreateDefaultAlignedStore(Result, Ops[2]);

    Result = Builder.CreateExtractValue(Call, 1);
    Result = EmitX86MaskedCompareResult(*this, Result, NumElts, nullptr);
    return Builder.CreateDefaultAlignedStore(Result, Ops[3]);
  }

  case X86::BI__builtin_ia32_vpmultishiftqb128:
  case X86::BI__builtin_ia32_vpmultishiftqb256:
  case X86::BI__builtin_ia32_vpmultishiftqb512: {
    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_vpmultishiftqb128:
      ID = Intrinsic::x86_avx512_pmultishift_qb_128;
      break;
    case X86::BI__builtin_ia32_vpmultishiftqb256:
      ID = Intrinsic::x86_avx512_pmultishift_qb_256;
      break;
    case X86::BI__builtin_ia32_vpmultishiftqb512:
      ID = Intrinsic::x86_avx512_pmultishift_qb_512;
      break;
    }

    return Builder.CreateCall(CGM.getIntrinsic(ID), Ops);
  }

  case X86::BI__builtin_ia32_vpshufbitqmb128_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb256_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb512_mask: {
    unsigned NumElts =
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
    Value *MaskIn = Ops[2];
    Ops.erase(&Ops[2]);

    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_vpshufbitqmb128_mask:
      ID = Intrinsic::x86_avx512_vpshufbitqmb_128;
      break;
    case X86::BI__builtin_ia32_vpshufbitqmb256_mask:
      ID = Intrinsic::x86_avx512_vpshufbitqmb_256;
      break;
    case X86::BI__builtin_ia32_vpshufbitqmb512_mask:
      ID = Intrinsic::x86_avx512_vpshufbitqmb_512;
      break;
    }

    Value *Shufbit = Builder.CreateCall(CGM.getIntrinsic(ID), Ops);
    return EmitX86MaskedCompareResult(*this, Shufbit, NumElts, MaskIn);
  }

  // packed comparison intrinsics
  case X86::BI__builtin_ia32_cmpeqps:
  case X86::BI__builtin_ia32_cmpeqpd:
    return getVectorFCmpIR(CmpInst::FCMP_OEQ, /*IsSignaling*/false);
  case X86::BI__builtin_ia32_cmpltps:
  case X86::BI__builtin_ia32_cmpltpd:
    return getVectorFCmpIR(CmpInst::FCMP_OLT, /*IsSignaling*/true);
  case X86::BI__builtin_ia32_cmpleps:
  case X86::BI__builtin_ia32_cmplepd:
    return getVectorFCmpIR(CmpInst::FCMP_OLE, /*IsSignaling*/true);
  case X86::BI__builtin_ia32_cmpunordps:
  case X86::BI__builtin_ia32_cmpunordpd:
    return getVectorFCmpIR(CmpInst::FCMP_UNO, /*IsSignaling*/false);
  case X86::BI__builtin_ia32_cmpneqps:
  case X86::BI__builtin_ia32_cmpneqpd:
    return getVectorFCmpIR(CmpInst::FCMP_UNE, /*IsSignaling*/false);
  case X86::BI__builtin_ia32_cmpnltps:
  case X86::BI__builtin_ia32_cmpnltpd:
    return getVectorFCmpIR(CmpInst::FCMP_UGE, /*IsSignaling*/true);
  case X86::BI__builtin_ia32_cmpnleps:
  case X86::BI__builtin_ia32_cmpnlepd:
    return getVectorFCmpIR(CmpInst::FCMP_UGT, /*IsSignaling*/true);
  case X86::BI__builtin_ia32_cmpordps:
  case X86::BI__builtin_ia32_cmpordpd:
    return getVectorFCmpIR(CmpInst::FCMP_ORD, /*IsSignaling*/false);
  case X86::BI__builtin_ia32_cmpph128_mask:
  case X86::BI__builtin_ia32_cmpph256_mask:
  case X86::BI__builtin_ia32_cmpph512_mask:
  case X86::BI__builtin_ia32_cmpps128_mask:
  case X86::BI__builtin_ia32_cmpps256_mask:
  case X86::BI__builtin_ia32_cmpps512_mask:
  case X86::BI__builtin_ia32_cmppd128_mask:
  case X86::BI__builtin_ia32_cmppd256_mask:
  case X86::BI__builtin_ia32_cmppd512_mask:
  case X86::BI__builtin_ia32_vcmpbf16512_mask:
  case X86::BI__builtin_ia32_vcmpbf16256_mask:
  case X86::BI__builtin_ia32_vcmpbf16128_mask:
    IsMaskFCmp = true;
    [[fallthrough]];
  case X86::BI__builtin_ia32_cmpps:
  case X86::BI__builtin_ia32_cmpps256:
  case X86::BI__builtin_ia32_cmppd:
  case X86::BI__builtin_ia32_cmppd256: {
    // Lowering vector comparisons to fcmp instructions, while
    // ignoring signalling behaviour requested
    // ignoring rounding mode requested
    // This is only possible if fp-model is not strict and FENV_ACCESS is off.

    // The third argument is the comparison condition, and integer in the
    // range [0, 31]
    unsigned CC = cast<llvm::ConstantInt>(Ops[2])->getZExtValue() & 0x1f;

    // Lowering to IR fcmp instruction.
    // Ignoring requested signaling behaviour,
    // e.g. both _CMP_GT_OS & _CMP_GT_OQ are translated to FCMP_OGT.
    FCmpInst::Predicate Pred;
    bool IsSignaling;
    // Predicates for 16-31 repeat the 0-15 predicates. Only the signalling
    // behavior is inverted. We'll handle that after the switch.
    switch (CC & 0xf) {
    case 0x00: Pred = FCmpInst::FCMP_OEQ;   IsSignaling = false; break;
    case 0x01: Pred = FCmpInst::FCMP_OLT;   IsSignaling = true;  break;
    case 0x02: Pred = FCmpInst::FCMP_OLE;   IsSignaling = true;  break;
    case 0x03: Pred = FCmpInst::FCMP_UNO;   IsSignaling = false; break;
    case 0x04: Pred = FCmpInst::FCMP_UNE;   IsSignaling = false; break;
    case 0x05: Pred = FCmpInst::FCMP_UGE;   IsSignaling = true;  break;
    case 0x06: Pred = FCmpInst::FCMP_UGT;   IsSignaling = true;  break;
    case 0x07: Pred = FCmpInst::FCMP_ORD;   IsSignaling = false; break;
    case 0x08: Pred = FCmpInst::FCMP_UEQ;   IsSignaling = false; break;
    case 0x09: Pred = FCmpInst::FCMP_ULT;   IsSignaling = true;  break;
    case 0x0a: Pred = FCmpInst::FCMP_ULE;   IsSignaling = true;  break;
    case 0x0b: Pred = FCmpInst::FCMP_FALSE; IsSignaling = false; break;
    case 0x0c: Pred = FCmpInst::FCMP_ONE;   IsSignaling = false; break;
    case 0x0d: Pred = FCmpInst::FCMP_OGE;   IsSignaling = true;  break;
    case 0x0e: Pred = FCmpInst::FCMP_OGT;   IsSignaling = true;  break;
    case 0x0f: Pred = FCmpInst::FCMP_TRUE;  IsSignaling = false; break;
    default: llvm_unreachable("Unhandled CC");
    }

    // Invert the signalling behavior for 16-31.
    if (CC & 0x10)
      IsSignaling = !IsSignaling;

    // If the predicate is true or false and we're using constrained intrinsics,
    // we don't have a compare intrinsic we can use. Just use the legacy X86
    // specific intrinsic.
    // If the intrinsic is mask enabled and we're using constrained intrinsics,
    // use the legacy X86 specific intrinsic.
    if (Builder.getIsFPConstrained() &&
        (Pred == FCmpInst::FCMP_TRUE || Pred == FCmpInst::FCMP_FALSE ||
         IsMaskFCmp)) {

      Intrinsic::ID IID;
      switch (BuiltinID) {
      default: llvm_unreachable("Unexpected builtin");
      case X86::BI__builtin_ia32_cmpps:
        IID = Intrinsic::x86_sse_cmp_ps;
        break;
      case X86::BI__builtin_ia32_cmpps256:
        IID = Intrinsic::x86_avx_cmp_ps_256;
        break;
      case X86::BI__builtin_ia32_cmppd:
        IID = Intrinsic::x86_sse2_cmp_pd;
        break;
      case X86::BI__builtin_ia32_cmppd256:
        IID = Intrinsic::x86_avx_cmp_pd_256;
        break;
      case X86::BI__builtin_ia32_cmpph128_mask:
        IID = Intrinsic::x86_avx512fp16_mask_cmp_ph_128;
        break;
      case X86::BI__builtin_ia32_cmpph256_mask:
        IID = Intrinsic::x86_avx512fp16_mask_cmp_ph_256;
        break;
      case X86::BI__builtin_ia32_cmpph512_mask:
        IID = Intrinsic::x86_avx512fp16_mask_cmp_ph_512;
        break;
      case X86::BI__builtin_ia32_cmpps512_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_ps_512;
        break;
      case X86::BI__builtin_ia32_cmppd512_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_pd_512;
        break;
      case X86::BI__builtin_ia32_cmpps128_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_ps_128;
        break;
      case X86::BI__builtin_ia32_cmpps256_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_ps_256;
        break;
      case X86::BI__builtin_ia32_cmppd128_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_pd_128;
        break;
      case X86::BI__builtin_ia32_cmppd256_mask:
        IID = Intrinsic::x86_avx512_mask_cmp_pd_256;
        break;
      }

      Function *Intr = CGM.getIntrinsic(IID);
      if (IsMaskFCmp) {
        unsigned NumElts =
            cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
        Ops[3] = getMaskVecValue(*this, Ops[3], NumElts);
        Value *Cmp = Builder.CreateCall(Intr, Ops);
        return EmitX86MaskedCompareResult(*this, Cmp, NumElts, nullptr);
      }

      return Builder.CreateCall(Intr, Ops);
    }

    // Builtins without the _mask suffix return a vector of integers
    // of the same width as the input vectors
    if (IsMaskFCmp) {
      // We ignore SAE if strict FP is disabled. We only keep precise
      // exception behavior under strict FP.
      // NOTE: If strict FP does ever go through here a CGFPOptionsRAII
      // object will be required.
      unsigned NumElts =
          cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements();
      Value *Cmp;
      if (IsSignaling)
        Cmp = Builder.CreateFCmpS(Pred, Ops[0], Ops[1]);
      else
        Cmp = Builder.CreateFCmp(Pred, Ops[0], Ops[1]);
      return EmitX86MaskedCompareResult(*this, Cmp, NumElts, Ops[3]);
    }

    return getVectorFCmpIR(Pred, IsSignaling);
  }

  // SSE scalar comparison intrinsics
  case X86::BI__builtin_ia32_cmpeqss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 0);
  case X86::BI__builtin_ia32_cmpltss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 1);
  case X86::BI__builtin_ia32_cmpless:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 2);
  case X86::BI__builtin_ia32_cmpunordss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 3);
  case X86::BI__builtin_ia32_cmpneqss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 4);
  case X86::BI__builtin_ia32_cmpnltss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 5);
  case X86::BI__builtin_ia32_cmpnless:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 6);
  case X86::BI__builtin_ia32_cmpordss:
    return getCmpIntrinsicCall(Intrinsic::x86_sse_cmp_ss, 7);
  case X86::BI__builtin_ia32_cmpeqsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 0);
  case X86::BI__builtin_ia32_cmpltsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 1);
  case X86::BI__builtin_ia32_cmplesd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 2);
  case X86::BI__builtin_ia32_cmpunordsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 3);
  case X86::BI__builtin_ia32_cmpneqsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 4);
  case X86::BI__builtin_ia32_cmpnltsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 5);
  case X86::BI__builtin_ia32_cmpnlesd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 6);
  case X86::BI__builtin_ia32_cmpordsd:
    return getCmpIntrinsicCall(Intrinsic::x86_sse2_cmp_sd, 7);

  // f16c half2float intrinsics
  case X86::BI__builtin_ia32_vcvtph2ps_mask:
  case X86::BI__builtin_ia32_vcvtph2ps256_mask:
  case X86::BI__builtin_ia32_vcvtph2ps512_mask: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    return EmitX86CvtF16ToFloatExpr(*this, Ops, ConvertType(E->getType()));
  }

  // AVX512 bf16 intrinsics
  case X86::BI__builtin_ia32_cvtneps2bf16_128_mask: {
    Ops[2] = getMaskVecValue(
        *this, Ops[2],
        cast<llvm::FixedVectorType>(Ops[0]->getType())->getNumElements());
    Intrinsic::ID IID = Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128;
    return Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
  }
  case X86::BI__builtin_ia32_cvtsbf162ss_32:
    return Builder.CreateFPExt(Ops[0], Builder.getFloatTy());

  case X86::BI__builtin_ia32_cvtneps2bf16_256_mask:
  case X86::BI__builtin_ia32_cvtneps2bf16_512_mask: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_cvtneps2bf16_256_mask:
      IID = Intrinsic::x86_avx512bf16_cvtneps2bf16_256;
      break;
    case X86::BI__builtin_ia32_cvtneps2bf16_512_mask:
      IID = Intrinsic::x86_avx512bf16_cvtneps2bf16_512;
      break;
    }
    Value *Res = Builder.CreateCall(CGM.getIntrinsic(IID), Ops[0]);
    return EmitX86Select(*this, Ops[2], Res, Ops[1]);
  }

  case X86::BI__cpuid:
  case X86::BI__cpuidex: {
    Value *FuncId = EmitScalarExpr(E->getArg(1));
    Value *SubFuncId = BuiltinID == X86::BI__cpuidex
                           ? EmitScalarExpr(E->getArg(2))
                           : llvm::ConstantInt::get(Int32Ty, 0);

    llvm::StructType *CpuidRetTy =
        llvm::StructType::get(Int32Ty, Int32Ty, Int32Ty, Int32Ty);
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(CpuidRetTy, {Int32Ty, Int32Ty}, false);

    StringRef Asm, Constraints;
    if (getTarget().getTriple().getArch() == llvm::Triple::x86) {
      Asm = "cpuid";
      Constraints = "={ax},={bx},={cx},={dx},{ax},{cx}";
    } else {
      // x86-64 uses %rbx as the base register, so preserve it.
      Asm = "xchgq %rbx, ${1:q}\n"
            "cpuid\n"
            "xchgq %rbx, ${1:q}";
      Constraints = "={ax},=r,={cx},={dx},0,2";
    }

    llvm::InlineAsm *IA = llvm::InlineAsm::get(FTy, Asm, Constraints,
                                               /*hasSideEffects=*/false);
    Value *IACall = Builder.CreateCall(IA, {FuncId, SubFuncId});
    Value *BasePtr = EmitScalarExpr(E->getArg(0));
    Value *Store = nullptr;
    for (unsigned i = 0; i < 4; i++) {
      Value *Extracted = Builder.CreateExtractValue(IACall, i);
      Value *StorePtr = Builder.CreateConstInBoundsGEP1_32(Int32Ty, BasePtr, i);
      Store = Builder.CreateAlignedStore(Extracted, StorePtr, getIntAlign());
    }

    // Return the last store instruction to signal that we have emitted the
    // the intrinsic.
    return Store;
  }

  case X86::BI__emul:
  case X86::BI__emulu: {
    llvm::Type *Int64Ty = llvm::IntegerType::get(getLLVMContext(), 64);
    bool isSigned = (BuiltinID == X86::BI__emul);
    Value *LHS = Builder.CreateIntCast(Ops[0], Int64Ty, isSigned);
    Value *RHS = Builder.CreateIntCast(Ops[1], Int64Ty, isSigned);
    return Builder.CreateMul(LHS, RHS, "", !isSigned, isSigned);
  }
  case X86::BI__mulh:
  case X86::BI__umulh:
  case X86::BI_mul128:
  case X86::BI_umul128: {
    llvm::Type *ResType = ConvertType(E->getType());
    llvm::Type *Int128Ty = llvm::IntegerType::get(getLLVMContext(), 128);

    bool IsSigned = (BuiltinID == X86::BI__mulh || BuiltinID == X86::BI_mul128);
    Value *LHS = Builder.CreateIntCast(Ops[0], Int128Ty, IsSigned);
    Value *RHS = Builder.CreateIntCast(Ops[1], Int128Ty, IsSigned);

    Value *MulResult, *HigherBits;
    if (IsSigned) {
      MulResult = Builder.CreateNSWMul(LHS, RHS);
      HigherBits = Builder.CreateAShr(MulResult, 64);
    } else {
      MulResult = Builder.CreateNUWMul(LHS, RHS);
      HigherBits = Builder.CreateLShr(MulResult, 64);
    }
    HigherBits = Builder.CreateIntCast(HigherBits, ResType, IsSigned);

    if (BuiltinID == X86::BI__mulh || BuiltinID == X86::BI__umulh)
      return HigherBits;

    Address HighBitsAddress = EmitPointerWithAlignment(E->getArg(2));
    Builder.CreateStore(HigherBits, HighBitsAddress);
    return Builder.CreateIntCast(MulResult, ResType, IsSigned);
  }

  case X86::BI__faststorefence: {
    return Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                               llvm::SyncScope::System);
  }
  case X86::BI__shiftleft128:
  case X86::BI__shiftright128: {
    llvm::Function *F = CGM.getIntrinsic(
        BuiltinID == X86::BI__shiftleft128 ? Intrinsic::fshl : Intrinsic::fshr,
        Int64Ty);
    // Flip low/high ops and zero-extend amount to matching type.
    // shiftleft128(Low, High, Amt) -> fshl(High, Low, Amt)
    // shiftright128(Low, High, Amt) -> fshr(High, Low, Amt)
    std::swap(Ops[0], Ops[1]);
    Ops[2] = Builder.CreateZExt(Ops[2], Int64Ty);
    return Builder.CreateCall(F, Ops);
  }
  case X86::BI_ReadWriteBarrier:
  case X86::BI_ReadBarrier:
  case X86::BI_WriteBarrier: {
    return Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                               llvm::SyncScope::SingleThread);
  }

  case X86::BI_AddressOfReturnAddress: {
    Function *F =
        CGM.getIntrinsic(Intrinsic::addressofreturnaddress, AllocaInt8PtrTy);
    return Builder.CreateCall(F);
  }
  case X86::BI__stosb: {
    // We treat __stosb as a volatile memset - it may not generate "rep stosb"
    // instruction, but it will create a memset that won't be optimized away.
    return Builder.CreateMemSet(Ops[0], Ops[1], Ops[2], Align(1), true);
  }
  // Corresponding to intrisics which will return 2 tiles (tile0_tile1).
  case X86::BI__builtin_ia32_t2rpntlvwz0_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0rs_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0t1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0rst1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1rs_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1t1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1rst1_internal: {
    Intrinsic::ID IID;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_t2rpntlvwz0_internal:
      IID = Intrinsic::x86_t2rpntlvwz0_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz0rs_internal:
      IID = Intrinsic::x86_t2rpntlvwz0rs_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz0t1_internal:
      IID = Intrinsic::x86_t2rpntlvwz0t1_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz0rst1_internal:
      IID = Intrinsic::x86_t2rpntlvwz0rst1_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz1_internal:
      IID = Intrinsic::x86_t2rpntlvwz1_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz1rs_internal:
      IID = Intrinsic::x86_t2rpntlvwz1rs_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz1t1_internal:
      IID = Intrinsic::x86_t2rpntlvwz1t1_internal;
      break;
    case X86::BI__builtin_ia32_t2rpntlvwz1rst1_internal:
      IID = Intrinsic::x86_t2rpntlvwz1rst1_internal;
      break;
    }

    // Ops = (Row0, Col0, Col1, DstPtr0, DstPtr1, SrcPtr, Stride)
    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID),
                                     {Ops[0], Ops[1], Ops[2], Ops[5], Ops[6]});

    auto *PtrTy = E->getArg(3)->getType()->getAs<PointerType>();
    assert(PtrTy && "arg3 must be of pointer type");
    QualType PtreeTy = PtrTy->getPointeeType();
    llvm::Type *TyPtee = ConvertType(PtreeTy);

    // Bitcast amx type (x86_amx) to vector type (256 x i32)
    // Then store tile0 into DstPtr0
    Value *T0 = Builder.CreateExtractValue(Call, 0);
    Value *VecT0 = Builder.CreateIntrinsic(Intrinsic::x86_cast_tile_to_vector,
                                           {TyPtee}, {T0});
    Builder.CreateDefaultAlignedStore(VecT0, Ops[3]);

    // Then store tile1 into DstPtr1
    Value *T1 = Builder.CreateExtractValue(Call, 1);
    Value *VecT1 = Builder.CreateIntrinsic(Intrinsic::x86_cast_tile_to_vector,
                                           {TyPtee}, {T1});
    Value *Store = Builder.CreateDefaultAlignedStore(VecT1, Ops[4]);

    // Note: Here we escape directly use x86_tilestored64_internal to store
    // the results due to it can't make sure the Mem written scope. This may
    // cause shapes reloads after first amx intrinsic, which current amx reg-
    // ister allocation has no ability to handle it.

    return Store;
  }
  case X86::BI__ud2:
    // llvm.trap makes a ud2a instruction on x86.
    return EmitTrapCall(Intrinsic::trap);
  case X86::BI__int2c: {
    // This syscall signals a driver assertion failure in x86 NT kernels.
    llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, false);
    llvm::InlineAsm *IA =
        llvm::InlineAsm::get(FTy, "int $$0x2c", "", /*hasSideEffects=*/true);
    llvm::AttributeList NoReturnAttr = llvm::AttributeList::get(
        getLLVMContext(), llvm::AttributeList::FunctionIndex,
        llvm::Attribute::NoReturn);
    llvm::CallInst *CI = Builder.CreateCall(IA);
    CI->setAttributes(NoReturnAttr);
    return CI;
  }
  case X86::BI__readfsbyte:
  case X86::BI__readfsword:
  case X86::BI__readfsdword:
  case X86::BI__readfsqword: {
    llvm::Type *IntTy = ConvertType(E->getType());
    Value *Ptr = Builder.CreateIntToPtr(
        Ops[0], llvm::PointerType::get(getLLVMContext(), 257));
    LoadInst *Load = Builder.CreateAlignedLoad(
        IntTy, Ptr, getContext().getTypeAlignInChars(E->getType()));
    Load->setVolatile(true);
    return Load;
  }
  case X86::BI__readgsbyte:
  case X86::BI__readgsword:
  case X86::BI__readgsdword:
  case X86::BI__readgsqword: {
    llvm::Type *IntTy = ConvertType(E->getType());
    Value *Ptr = Builder.CreateIntToPtr(
        Ops[0], llvm::PointerType::get(getLLVMContext(), 256));
    LoadInst *Load = Builder.CreateAlignedLoad(
        IntTy, Ptr, getContext().getTypeAlignInChars(E->getType()));
    Load->setVolatile(true);
    return Load;
  }
  case X86::BI__builtin_ia32_encodekey128_u32: {
    Intrinsic::ID IID = Intrinsic::x86_encodekey128;

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), {Ops[0], Ops[1]});

    for (int i = 0; i < 3; ++i) {
      Value *Extract = Builder.CreateExtractValue(Call, i + 1);
      Value *Ptr = Builder.CreateConstGEP1_32(Int8Ty, Ops[2], i * 16);
      Builder.CreateAlignedStore(Extract, Ptr, Align(1));
    }

    return Builder.CreateExtractValue(Call, 0);
  }
  case X86::BI__builtin_ia32_encodekey256_u32: {
    Intrinsic::ID IID = Intrinsic::x86_encodekey256;

    Value *Call =
        Builder.CreateCall(CGM.getIntrinsic(IID), {Ops[0], Ops[1], Ops[2]});

    for (int i = 0; i < 4; ++i) {
      Value *Extract = Builder.CreateExtractValue(Call, i + 1);
      Value *Ptr = Builder.CreateConstGEP1_32(Int8Ty, Ops[3], i * 16);
      Builder.CreateAlignedStore(Extract, Ptr, Align(1));
    }

    return Builder.CreateExtractValue(Call, 0);
  }
  case X86::BI__builtin_ia32_aesenc128kl_u8:
  case X86::BI__builtin_ia32_aesdec128kl_u8:
  case X86::BI__builtin_ia32_aesenc256kl_u8:
  case X86::BI__builtin_ia32_aesdec256kl_u8: {
    Intrinsic::ID IID;
    StringRef BlockName;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_aesenc128kl_u8:
      IID = Intrinsic::x86_aesenc128kl;
      BlockName = "aesenc128kl";
      break;
    case X86::BI__builtin_ia32_aesdec128kl_u8:
      IID = Intrinsic::x86_aesdec128kl;
      BlockName = "aesdec128kl";
      break;
    case X86::BI__builtin_ia32_aesenc256kl_u8:
      IID = Intrinsic::x86_aesenc256kl;
      BlockName = "aesenc256kl";
      break;
    case X86::BI__builtin_ia32_aesdec256kl_u8:
      IID = Intrinsic::x86_aesdec256kl;
      BlockName = "aesdec256kl";
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), {Ops[1], Ops[2]});

    BasicBlock *NoError =
        createBasicBlock(BlockName + "_no_error", this->CurFn);
    BasicBlock *Error = createBasicBlock(BlockName + "_error", this->CurFn);
    BasicBlock *End = createBasicBlock(BlockName + "_end", this->CurFn);

    Value *Ret = Builder.CreateExtractValue(Call, 0);
    Value *Succ = Builder.CreateTrunc(Ret, Builder.getInt1Ty());
    Value *Out = Builder.CreateExtractValue(Call, 1);
    Builder.CreateCondBr(Succ, NoError, Error);

    Builder.SetInsertPoint(NoError);
    Builder.CreateDefaultAlignedStore(Out, Ops[0]);
    Builder.CreateBr(End);

    Builder.SetInsertPoint(Error);
    Constant *Zero = llvm::Constant::getNullValue(Out->getType());
    Builder.CreateDefaultAlignedStore(Zero, Ops[0]);
    Builder.CreateBr(End);

    Builder.SetInsertPoint(End);
    return Builder.CreateExtractValue(Call, 0);
  }
  case X86::BI__builtin_ia32_aesencwide128kl_u8:
  case X86::BI__builtin_ia32_aesdecwide128kl_u8:
  case X86::BI__builtin_ia32_aesencwide256kl_u8:
  case X86::BI__builtin_ia32_aesdecwide256kl_u8: {
    Intrinsic::ID IID;
    StringRef BlockName;
    switch (BuiltinID) {
    case X86::BI__builtin_ia32_aesencwide128kl_u8:
      IID = Intrinsic::x86_aesencwide128kl;
      BlockName = "aesencwide128kl";
      break;
    case X86::BI__builtin_ia32_aesdecwide128kl_u8:
      IID = Intrinsic::x86_aesdecwide128kl;
      BlockName = "aesdecwide128kl";
      break;
    case X86::BI__builtin_ia32_aesencwide256kl_u8:
      IID = Intrinsic::x86_aesencwide256kl;
      BlockName = "aesencwide256kl";
      break;
    case X86::BI__builtin_ia32_aesdecwide256kl_u8:
      IID = Intrinsic::x86_aesdecwide256kl;
      BlockName = "aesdecwide256kl";
      break;
    }

    llvm::Type *Ty = FixedVectorType::get(Builder.getInt64Ty(), 2);
    Value *InOps[9];
    InOps[0] = Ops[2];
    for (int i = 0; i != 8; ++i) {
      Value *Ptr = Builder.CreateConstGEP1_32(Ty, Ops[1], i);
      InOps[i + 1] = Builder.CreateAlignedLoad(Ty, Ptr, Align(16));
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), InOps);

    BasicBlock *NoError =
        createBasicBlock(BlockName + "_no_error", this->CurFn);
    BasicBlock *Error = createBasicBlock(BlockName + "_error", this->CurFn);
    BasicBlock *End = createBasicBlock(BlockName + "_end", this->CurFn);

    Value *Ret = Builder.CreateExtractValue(Call, 0);
    Value *Succ = Builder.CreateTrunc(Ret, Builder.getInt1Ty());
    Builder.CreateCondBr(Succ, NoError, Error);

    Builder.SetInsertPoint(NoError);
    for (int i = 0; i != 8; ++i) {
      Value *Extract = Builder.CreateExtractValue(Call, i + 1);
      Value *Ptr = Builder.CreateConstGEP1_32(Extract->getType(), Ops[0], i);
      Builder.CreateAlignedStore(Extract, Ptr, Align(16));
    }
    Builder.CreateBr(End);

    Builder.SetInsertPoint(Error);
    for (int i = 0; i != 8; ++i) {
      Value *Out = Builder.CreateExtractValue(Call, i + 1);
      Constant *Zero = llvm::Constant::getNullValue(Out->getType());
      Value *Ptr = Builder.CreateConstGEP1_32(Out->getType(), Ops[0], i);
      Builder.CreateAlignedStore(Zero, Ptr, Align(16));
    }
    Builder.CreateBr(End);

    Builder.SetInsertPoint(End);
    return Builder.CreateExtractValue(Call, 0);
  }
  case X86::BI__builtin_ia32_vfcmaddcph512_mask:
    IsConjFMA = true;
    [[fallthrough]];
  case X86::BI__builtin_ia32_vfmaddcph512_mask: {
    Intrinsic::ID IID = IsConjFMA
                            ? Intrinsic::x86_avx512fp16_mask_vfcmadd_cph_512
                            : Intrinsic::x86_avx512fp16_mask_vfmadd_cph_512;
    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
    return EmitX86Select(*this, Ops[3], Call, Ops[0]);
  }
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask:
    IsConjFMA = true;
    [[fallthrough]];
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask: {
    Intrinsic::ID IID = IsConjFMA ? Intrinsic::x86_avx512fp16_mask_vfcmadd_csh
                                  : Intrinsic::x86_avx512fp16_mask_vfmadd_csh;
    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
    Value *And = Builder.CreateAnd(Ops[3], llvm::ConstantInt::get(Int8Ty, 1));
    return EmitX86Select(*this, And, Call, Ops[0]);
  }
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask3:
    IsConjFMA = true;
    [[fallthrough]];
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask3: {
    Intrinsic::ID IID = IsConjFMA ? Intrinsic::x86_avx512fp16_mask_vfcmadd_csh
                                  : Intrinsic::x86_avx512fp16_mask_vfmadd_csh;
    Value *Call = Builder.CreateCall(CGM.getIntrinsic(IID), Ops);
    static constexpr int Mask[] = {0, 5, 6, 7};
    return Builder.CreateShuffleVector(Call, Ops[2], Mask);
  }
  case X86::BI__builtin_ia32_prefetchi:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::prefetch, Ops[0]->getType()),
        {Ops[0], llvm::ConstantInt::get(Int32Ty, 0), Ops[1],
         llvm::ConstantInt::get(Int32Ty, 0)});
  }
}
