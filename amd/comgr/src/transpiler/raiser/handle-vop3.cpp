//===- handle-vop3.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/handle-vop-cross-lane.h"
#include "transpiler/raiser/handle-vop-shared.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <climits>
#include <cstdint>

using namespace llvm;

namespace COMGR::transpiler {
namespace {

Value *isNaNBits(IRBuilder<> &B, Value *Float) {
  Value *Bits = B.CreateBitCast(Float, B.getInt32Ty());
  Value *Magnitude = B.CreateAnd(Bits, B.getInt32(0x7fffffff));
  return B.CreateICmpUGT(Magnitude, B.getInt32(0x7f800000));
}

Value *quietSignalingNaN(IRBuilder<> &B, Value *Source, Value *Fallback) {
  Value *Bits = B.CreateBitCast(Source, B.getInt32Ty());
  Value *IsNaN = isNaNBits(B, Source);
  Value *QuietBit = B.CreateAnd(Bits, B.getInt32(0x00400000));
  Value *IsSignaling = B.CreateICmpEQ(QuietBit, B.getInt32(0));
  Value *SelectSource = B.CreateAnd(IsNaN, IsSignaling);
  Value *QuietedBits = B.CreateOr(Bits, B.getInt32(0x00400000));
  Value *Quieted = B.CreateBitCast(QuietedBits, B.getFloatTy());
  return B.CreateSelect(SelectSource, Quieted, Fallback);
}

// Older IEEE-mode extrema quiet a signaling NaN before the next operation.
Value *numericExtremum(IRBuilder<> &B, Intrinsic::ID ID, Value *Src0,
                       Value *Src1, bool LegacyIeee) {
  Value *Result = B.CreateBinaryIntrinsic(ID, Src0, Src1);
  if (!LegacyIeee)
    return Result;
  Result = quietSignalingNaN(B, Src1, Result);
  return quietSignalingNaN(B, Src0, Result);
}

/// Reject nonzero output multipliers on integer VOP3 instructions.
Error requireNoOutputMultiplier(RaiseContext &Ctx, const DecodedInst &Di) {
  int Idx = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                  AMDGPU::OpName::omod);
  if (Idx < 0)
    return Error::success();
  if (!Di.isImm(Idx))
    return unsupportedInstruction(Ctx, Di, "omod operand is not immediate");
  if (Di.getImm(Idx) != 0)
    return unsupportedInstruction(Ctx, Di,
                                  "integer output multiplier is not supported");
  return Error::success();
}

/// Reject non-default floating-point output modifiers.
Error requireDefaultFloatOutputModifiers(RaiseContext &Ctx,
                                         const DecodedInst &Di) {
  int OmodIndex = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                        AMDGPU::OpName::omod);
  if (OmodIndex >= 0) {
    if (!Di.isImm(OmodIndex))
      return unsupportedInstruction(Ctx, Di, "omod operand is not immediate");
    if (Di.getImm(OmodIndex) != 0)
      return unsupportedInstruction(
          Ctx, Di, "floating-point output multiplier is not supported");
  }
  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();
  if (*Clamp)
    return unsupportedInstruction(
        Ctx, Di, "floating-point output clamp is not supported");
  return Error::success();
}

/// Reject source modifiers, which have floating-point rather than integer
/// semantics.
Error requireNoIntegerSourceModifiers(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op) {
  const unsigned NumSources = Op.nSrcs();
  for (unsigned I = 0; I != NumSources; ++I) {
    if (Op.srcMod(I) != 0)
      return unsupportedInstruction(
          Ctx, Di, "integer source modifiers are not supported");
  }
  return Error::success();
}

/// Raise a wrapping binary integer operation at the selected width.
Error handleBinary(RaiseContext &Ctx, OperandResolver &Op,
                   Instruction::BinaryOps Opcode, bool Is64,
                   bool Reverse = false) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0, Is64);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1, Is64);
  if (!Src1)
    return Src1.takeError();
  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *Result = Ctx.B.CreateBinOp(Opcode, Lhs, Rhs);
  if (Is64)
    Ctx.registers().writeReg64(*Dst, Result);
  else
    Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

/// Write a per-lane carry or borrow result to the instruction's second
/// destination.
Error writeCarryOut(RaiseContext &Ctx, const DecodedInst &Di,
                    OperandResolver &Op, Value *Carry) {
  if (Di.NumDefs != 2 || !Di.isReg(1))
    return unsupportedInstruction(Ctx, Di,
                                  "expected an explicit carry destination");
  Expected<ParsedReg> Dst = Op.dst(1);
  if (!Dst)
    return Dst.takeError();

  Carry = Ctx.B.CreateSelect(Ctx.registers().emitLaneActiveBit(), Carry,
                             Ctx.B.getFalse(), "carry.active");
  switch (Dst->RegKind) {
  case ParsedReg::SGPR: {
    Type *MaskTy = Ctx.Projection.sourceWaveMaskTy();
    Value *Mask =
        Ctx.Projection.ballotI1ToWidth(Ctx.B, Carry, MaskTy, "carry.mask");
    Ctx.registers().writeRegExecWidth(*Dst, Mask);
    Ctx.registers().recordWaveMaskI1(*Dst, Carry);
    return Error::success();
  }
  case ParsedReg::VCC:
    Ctx.registers().regFile().storeVCC(Ctx.B, Carry);
    return Error::success();
  case ParsedReg::NOREG:
    return Error::success();
  default:
    return unsupportedInstruction(Ctx, Di,
                                  "invalid carry destination register");
  }
}

/// Raise a binary integer operation with an explicit carry or borrow output.
Error handleBinaryCarry(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op, Intrinsic::ID OverflowIntrinsic,
                        Intrinsic::ID SaturatingIntrinsic, bool Reverse,
                        bool Clamp) {
  if (Di.NumDefs != 2 || Op.nSrcs() < 2)
    return unsupportedInstruction(Ctx, Di,
                                  "expected two destinations and two sources");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *Pair = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                      {Lhs, Rhs});
  Value *Wrapped = Ctx.B.CreateExtractValue(Pair, 0, "result");
  Value *Carry = Ctx.B.CreateExtractValue(Pair, 1, "carry");
  Value *Result =
      Clamp ? Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, Lhs, Rhs)
            : Wrapped;
  Ctx.registers().writeReg32(*Dst, Result);
  return writeCarryOut(Ctx, Di, Op, Carry);
}

/// Raise a binary integer operation with explicit carry or borrow input and
/// output.
Error handleBinaryCarryIn(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op, Intrinsic::ID OverflowIntrinsic,
                          Intrinsic::ID SaturatingIntrinsic, bool Reverse,
                          bool Clamp) {
  if (Di.NumDefs != 2 || Op.nSrcs() < 3)
    return unsupportedInstruction(
        Ctx, Di, "expected two destinations and three sources");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> CarryInI1 = Op.srcWaveMaskI1(2);
  if (!CarryInI1)
    return CarryInI1.takeError();
  if (!*CarryInI1)
    return unsupportedInstruction(Ctx, Di, "invalid carry-in register");

  Value *Lhs = Reverse ? *Src1 : *Src0;
  Value *Rhs = Reverse ? *Src0 : *Src1;
  Value *CarryIn = Ctx.B.CreateZExt(*CarryInI1, Ctx.B.getInt32Ty(), "carry.in");
  Value *First = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                       {Lhs, Rhs});
  Value *FirstResult = Ctx.B.CreateExtractValue(First, 0);
  Value *FirstCarry = Ctx.B.CreateExtractValue(First, 1);
  Value *Second = Ctx.B.CreateIntrinsic(OverflowIntrinsic, {Ctx.B.getInt32Ty()},
                                        {FirstResult, CarryIn});
  Value *Wrapped = Ctx.B.CreateExtractValue(Second, 0, "result");
  Value *SecondCarry = Ctx.B.CreateExtractValue(Second, 1);
  Value *Carry = Ctx.B.CreateOr(FirstCarry, SecondCarry, "carry");
  Value *Result = Wrapped;
  if (Clamp) {
    Value *FirstSat =
        Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, Lhs, Rhs);
    Result =
        Ctx.B.CreateBinaryIntrinsic(SaturatingIntrinsic, FirstSat, CarryIn);
  }
  Ctx.registers().writeReg32(*Dst, Result);
  return writeCarryOut(Ctx, Di, Op, Carry);
}

/// Extend the low 24 bits of V to Ty according to the instruction's signedness.
Value *extendLow24(IRBuilder<> &B, Value *V, Type *Ty, bool IsSigned) {
  Value *Low = B.CreateTrunc(V, B.getIntNTy(24), "low24");
  return IsSigned ? B.CreateSExt(Low, Ty, "sext24")
                  : B.CreateZExt(Low, Ty, "zext24");
}

/// Extract the byte at ByteIndex from a packed 64-bit value.
Value *extractByte(IRBuilder<> &B, Value *Packed, Value *ByteIndex,
                   const Twine &Name) {
  constexpr unsigned ByteOffsetShift = 3;
  Value *BitOffset32 =
      B.CreateShl(ByteIndex, B.getInt32(ByteOffsetShift), Name + ".offset");
  Value *BitOffset =
      B.CreateZExt(BitOffset32, B.getInt64Ty(), Name + ".offset64");
  Value *Shifted = B.CreateLShr(Packed, BitOffset, Name + ".shifted");
  return B.CreateTrunc(Shifted, B.getInt8Ty(), Name);
}

/// Select one byte from Packed according to the V_PERM_B32 selector encoding.
/// Selectors 0-7 choose an input byte, 8-11 replicate a sign bit, 12 produces
/// zero, and larger selectors produce all ones.
Value *permuteByte(IRBuilder<> &B, Value *Packed, Value *Selector) {
  constexpr unsigned DataSelectorMask = 7;
  constexpr unsigned FirstSignSelector = 8;
  constexpr unsigned NumSignSelectors = 4;
  constexpr unsigned ZeroSelector = 12;
  constexpr unsigned FirstOnesSelector = 13;

  Value *ByteIndex =
      B.CreateAnd(Selector, B.getInt32(DataSelectorMask), "perm.index");
  Value *DataByte = extractByte(B, Packed, ByteIndex, "perm.data");

  Value *SignSelector =
      B.CreateSub(Selector, B.getInt32(FirstSignSelector), "perm.sign.sel");
  Value *IsSign = B.CreateICmpULT(SignSelector, B.getInt32(NumSignSelectors),
                                  "perm.is.sign");
  Value *SignIndex =
      B.CreateAnd(SignSelector, B.getInt32(NumSignSelectors - 1));
  Value *SignIndexTwice = B.CreateShl(SignIndex, B.getInt32(1));
  Value *SignByteIndex =
      B.CreateAdd(SignIndexTwice, B.getInt32(1), "perm.sign.index");
  Value *SignByte = extractByte(B, Packed, SignByteIndex, "perm.sign.byte");
  Value *SignBit = B.CreateICmpSLT(SignByte, B.getInt8(0), "perm.sign.bit");
  Value *SignFill =
      B.CreateSelect(SignBit, B.getInt8(-1), B.getInt8(0), "perm.sign.fill");
  Value *Result = B.CreateSelect(IsSign, SignFill, DataByte, "perm.sign");

  Value *IsZero =
      B.CreateICmpEQ(Selector, B.getInt32(ZeroSelector), "perm.is.zero");
  Result = B.CreateSelect(IsZero, B.getInt8(0), Result, "perm.zero");
  Value *IsOnes =
      B.CreateICmpUGE(Selector, B.getInt32(FirstOnesSelector), "perm.is.ones");
  return B.CreateSelect(IsOnes, B.getInt8(-1), Result, "perm.byte");
}

/// Raise V_PERM_B32 using generic integer operations.
Value *raiseBytePermute(IRBuilder<> &B, const TernaryOperands &Args) {
  constexpr unsigned BitsPerByte = 8;
  constexpr unsigned BytesPerDword = 4;
  constexpr unsigned HighDwordOffset = 32;
  constexpr unsigned SelectorMask = 0xff;

  Value *Low = B.CreateZExt(Args.Src1, B.getInt64Ty(), "perm.low");
  Value *High = B.CreateZExt(Args.Src0, B.getInt64Ty(), "perm.high");
  High = B.CreateShl(High, B.getInt64(HighDwordOffset), "perm.high.shifted");
  Value *Packed = B.CreateOr(High, Low, "perm.input");

  Value *Result = B.getInt32(0);
  for (unsigned I = 0; I != BytesPerDword; ++I) {
    Value *Selector = Args.Src2;
    if (I != 0)
      Selector =
          B.CreateLShr(Selector, B.getInt32(I * BitsPerByte), "perm.selector");
    Selector =
        B.CreateAnd(Selector, B.getInt32(SelectorMask), "perm.selector.byte");
    Value *Byte = permuteByte(B, Packed, Selector);
    Value *Extended = B.CreateZExt(Byte, B.getInt32Ty(), "perm.extended");
    if (I != 0)
      Extended =
          B.CreateShl(Extended, B.getInt32(I * BitsPerByte), "perm.positioned");
    Result = B.CreateOr(Result, Extended, "perm.result");
  }
  return Result;
}

/// Raise a nested min/max operation over three integer sources.
Error handleTernaryMinMax(RaiseContext &Ctx, OperandResolver &Op,
                          Intrinsic::ID Inner, Intrinsic::ID Outer,
                          StringRef Name) {
  Expected<TernaryOperands> Args = Op.readTernary32();
  if (!Args)
    return Args.takeError();
  Value *First = Ctx.B.CreateBinaryIntrinsic(Inner, Args->Src0, Args->Src1, {},
                                             Name + ".inner");
  Value *Result =
      Ctx.B.CreateBinaryIntrinsic(Outer, First, Args->Src2, {}, Name);
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

/// Raise an F32 or F64 ldexp with a floating-point significand and integer
/// exponent.
Error raiseLdexpFloat(RaiseContext &Ctx, const DecodedInst &Di,
                      OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 2)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and two sources");
  bool Is64 = Di.CanonOp == CanonicalOp::V_LDEXP_F64;
  Type *Ty = Is64 ? Ctx.B.getDoubleTy() : Ctx.B.getFloatTy();
  if (Error Err = Ctx.validateFPEnvironment(Di, Ty))
    return Err;
  if (Op.srcMod(1) != 0)
    return unsupportedInstruction(
        Ctx, Di, "integer exponent modifiers are not supported");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Significand = Is64 ? Op.srcF64(0) : Op.srcF32(0);
  if (!Significand)
    return Significand.takeError();
  Expected<Value *> Exponent = Op.src(1);
  if (!Exponent)
    return Exponent.takeError();
  Value *Result =
      Ctx.B.CreateIntrinsic(Intrinsic::ldexp, {Ty, Ctx.B.getInt32Ty()},
                            {*Significand, *Exponent}, nullptr, "ldexp");
  Type *BitsTy = Is64 ? Ctx.B.getInt64Ty() : Ctx.B.getInt32Ty();
  Value *ResultBits = Ctx.B.CreateBitCast(Result, BitsTy);
  if (Is64)
    Ctx.registers().writeReg64(*Dst, ResultBits);
  else
    Ctx.registers().writeReg32(*Dst, ResultBits);
  return Error::success();
}

Error raiseFloatTernary32(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op) {
  if (Di.NumDefs < 1 || Op.nSrcs() < 3)
    return unsupportedInstruction(Ctx, Di, "expected three float sources");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;
  if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
    return Err;

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  if (Di.CanonOp == CanonicalOp::V_DIV_SCALE_F32) {
    if (Di.NumDefs != 2 || !Di.isReg(1))
      return unsupportedInstruction(Ctx, Di,
                                    "expected a scale flag destination");
    Expected<ParsedReg> FlagDst = Op.dst(1);
    if (!FlagDst)
      return FlagDst.takeError();
    if (FlagDst->RegKind != ParsedReg::SGPR &&
        FlagDst->RegKind != ParsedReg::VCC &&
        FlagDst->RegKind != ParsedReg::NOREG)
      return unsupportedInstruction(Ctx, Di, "invalid scale flag destination");

    auto SameOperand = [&](unsigned A, unsigned B) {
      unsigned AI = Op.srcIdx(A), BI = Op.srcIdx(B);
      return (Di.isReg(AI) && Di.isReg(BI) && Di.getReg(AI) == Di.getReg(BI)) ||
             (Di.isImm(AI) && Di.isImm(BI) && Di.getImm(AI) == Di.getImm(BI));
    };
    bool NumeratorScale = SameOperand(0, 2);
    bool DenominatorScale = SameOperand(0, 1);
    if (!NumeratorScale && !DenominatorScale)
      return unsupportedInstruction(Ctx, Di,
                                    "unrecognized divide scale operand shape");
    if (NumeratorScale && DenominatorScale) {
      if (Op.srcMod(0) != Op.srcMod(1) || Op.srcMod(0) != Op.srcMod(2))
        return unsupportedInstruction(
            Ctx, Di, "asymmetric divide scale source modifiers");
      NumeratorScale = false;
    }
    if (Op.srcMod(0) != Op.srcMod(NumeratorScale ? 2 : 1))
      return unsupportedInstruction(Ctx, Di,
                                    "asymmetric divide scale source modifiers");

    Expected<Value *> Numer = Op.srcF32(NumeratorScale ? 0 : 2);
    if (!Numer)
      return Numer.takeError();
    Expected<Value *> Denom = Op.srcF32(1);
    if (!Denom)
      return Denom.takeError();
    Value *Pair = Ctx.B.CreateIntrinsic(
        Intrinsic::amdgcn_div_scale, {Ctx.B.getFloatTy()},
        {*Numer, *Denom, Ctx.B.getInt1(NumeratorScale)}, nullptr, "div.scale");
    Value *Result = Ctx.B.CreateExtractValue(Pair, 0);
    Value *Flag = Ctx.B.CreateExtractValue(Pair, 1);
    Ctx.registers().writeReg32(*Dst,
                               Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
    Flag = Ctx.B.CreateAnd(Flag, Ctx.registers().emitLaneActiveBit());
    if (FlagDst->RegKind == ParsedReg::SGPR) {
      Value *Mask = Ctx.Projection.ballotI1ToWidth(
          Ctx.B, Flag, Ctx.Projection.sourceWaveMaskTy(), "div.scale.mask");
      Ctx.registers().writeRegExecWidth(*FlagDst, Mask);
      Ctx.registers().recordWaveMaskI1(*FlagDst, Flag);
    } else if (FlagDst->RegKind == ParsedReg::VCC) {
      Ctx.registers().regFile().storeVCC(Ctx.B, Flag);
    }
    return Error::success();
  }

  Expected<Value *> Src0 = Op.srcF32(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.srcF32(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> Src2 = Op.srcF32(2);
  if (!Src2)
    return Src2.takeError();
  bool LegacyIeee = Ctx.Projection.SourceSTI.hasFeature(
                        AMDGPU::FeatureDX10ClampAndIEEEMode) &&
                    Ctx.sourceIeeeMode();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_DIV_FMAS_F32:
    Result = Ctx.B.CreateIntrinsic(
        Intrinsic::amdgcn_div_fmas, {Ctx.B.getFloatTy()},
        {*Src0, *Src1, *Src2, Ctx.registers().regFile().loadVCC(Ctx.B)},
        nullptr, "div.fmas");
    break;
  case CanonicalOp::V_DIV_FIXUP_F32:
    Result =
        Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_div_fixup, {Ctx.B.getFloatTy()},
                              {*Src0, *Src1, *Src2}, nullptr, "div.fixup");
    break;
  case CanonicalOp::V_MAX3_NUM_F32:
  case CanonicalOp::V_MIN3_NUM_F32: {
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_MAX3_NUM_F32
                           ? Intrinsic::maximumnum
                           : Intrinsic::minimumnum;
    Value *First = numericExtremum(Ctx.B, ID, *Src0, *Src1, LegacyIeee);
    Result = numericExtremum(Ctx.B, ID, First, *Src2, LegacyIeee);
    break;
  }
  case CanonicalOp::V_MED3_NUM_F32: {
    Result = Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_fmed3,
                                   {Ctx.B.getFloatTy()}, {*Src0, *Src1, *Src2});
    Value *FirstMin =
        numericExtremum(Ctx.B, Intrinsic::minimumnum, *Src0, *Src1, LegacyIeee);
    Value *NaNMin = numericExtremum(Ctx.B, Intrinsic::minimumnum, FirstMin,
                                    *Src2, LegacyIeee);
    Value *NaN0 = isNaNBits(Ctx.B, *Src0);
    Value *NaN1 = isNaNBits(Ctx.B, *Src1);
    Value *NaN2 = isNaNBits(Ctx.B, *Src2);
    Value *AnyNaN = Ctx.B.CreateOr(NaN0, NaN1);
    AnyNaN = Ctx.B.CreateOr(AnyNaN, NaN2);
    Result = Ctx.B.CreateSelect(AnyNaN, NaNMin, Result);
    break;
  }
  case CanonicalOp::V_MAXIMUM3_F32:
  case CanonicalOp::V_MINIMUM3_F32: {
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_MAXIMUM3_F32
                           ? Intrinsic::maximum
                           : Intrinsic::minimum;
    Value *First = Ctx.B.CreateBinaryIntrinsic(ID, *Src0, *Src1);
    Result = Ctx.B.CreateBinaryIntrinsic(ID, First, *Src2);
    break;
  }
  case CanonicalOp::V_MAXMIN_NUM_F32:
  case CanonicalOp::V_MINMAX_NUM_F32:
  case CanonicalOp::V_MAXIMUMMINIMUM_F32:
  case CanonicalOp::V_MINIMUMMAXIMUM_F32: {
    bool Numeric = Di.CanonOp == CanonicalOp::V_MAXMIN_NUM_F32 ||
                   Di.CanonOp == CanonicalOp::V_MINMAX_NUM_F32;
    bool MaxFirst = Di.CanonOp == CanonicalOp::V_MAXMIN_NUM_F32 ||
                    Di.CanonOp == CanonicalOp::V_MAXIMUMMINIMUM_F32;
    Intrinsic::ID MaxID = Numeric ? Intrinsic::maximumnum : Intrinsic::maximum;
    Intrinsic::ID MinID = Numeric ? Intrinsic::minimumnum : Intrinsic::minimum;
    Value *First =
        Ctx.B.CreateBinaryIntrinsic(MaxFirst ? MaxID : MinID, *Src0, *Src1);
    Result =
        Ctx.B.CreateBinaryIntrinsic(MaxFirst ? MinID : MaxID, First, *Src2);
    break;
  }
  default:
    llvm_unreachable("not a ternary F32 operation");
  }
  Ctx.registers().writeReg32(*Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

Error raiseFloatBinary32(RaiseContext &Ctx, const DecodedInst &Di,
                         OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 2)
    return unsupportedInstruction(Ctx, Di, "expected two float sources");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;
  if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
    return Err;
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = Op.srcF32(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.srcF32(1);
  if (!Src1)
    return Src1.takeError();
  Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_MAXIMUM_F32
                         ? Intrinsic::maximum
                         : Intrinsic::minimum;
  Value *Result = Ctx.B.CreateBinaryIntrinsic(ID, *Src0, *Src1);
  Ctx.registers().writeReg32(*Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

} // namespace

Error handleVOP3(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (std::optional<VectorCompareInfo> Info = getVectorCompareInfo(Di.CanonOp))
    return raiseVectorCompare(Ctx, Di, Op, *Info);

  switch (Di.CanonOp) {
  case CanonicalOp::V_NOP:
    return Error::success();
  case CanonicalOp::V_ADD_F32:
  case CanonicalOp::V_ADD_F64:
  case CanonicalOp::V_MUL_F32:
  case CanonicalOp::V_MUL_F64:
  case CanonicalOp::V_SUB_F32:
  case CanonicalOp::V_SUBREV_F32:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseFloatBinary(Ctx, Di, Op);
  case CanonicalOp::V_FMA_F32:
  case CanonicalOp::V_FMAC_F32:
  case CanonicalOp::V_FMA_F64:
  case CanonicalOp::V_FMAC_F64:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseFloatMac(Ctx, Di, Op);
  case CanonicalOp::V_MAX_NUM_F32:
  case CanonicalOp::V_MIN_NUM_F32:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseNumericExtremum32(Ctx, Di, Op);
  case CanonicalOp::V_MAXIMUM_F32:
  case CanonicalOp::V_MINIMUM_F32:
    return raiseFloatBinary32(Ctx, Di, Op);
  case CanonicalOp::V_CVT_F32_F64:
  case CanonicalOp::V_CVT_F64_F32:
  case CanonicalOp::V_CVT_F64_I32:
  case CanonicalOp::V_CVT_F64_U32:
  case CanonicalOp::V_CVT_I32_F64:
  case CanonicalOp::V_CVT_U32_F64:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseFloatConversion64(Ctx, Di, Op);
  case CanonicalOp::V_TRUNC_F64:
  case CanonicalOp::V_CEIL_F64:
  case CanonicalOp::V_RNDNE_F64:
  case CanonicalOp::V_FLOOR_F64:
  case CanonicalOp::V_RCP_F64:
  case CanonicalOp::V_RSQ_F64:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseUnaryFloat64(Ctx, Di, Op);
  case CanonicalOp::V_DIV_SCALE_F32:
  case CanonicalOp::V_DIV_FMAS_F32:
  case CanonicalOp::V_DIV_FIXUP_F32:
  case CanonicalOp::V_MAX3_NUM_F32:
  case CanonicalOp::V_MIN3_NUM_F32:
  case CanonicalOp::V_MED3_NUM_F32:
  case CanonicalOp::V_MAXIMUM3_F32:
  case CanonicalOp::V_MINIMUM3_F32:
  case CanonicalOp::V_MAXMIN_NUM_F32:
  case CanonicalOp::V_MINMAX_NUM_F32:
  case CanonicalOp::V_MAXIMUMMINIMUM_F32:
  case CanonicalOp::V_MINIMUMMAXIMUM_F32:
    return raiseFloatTernary32(Ctx, Di, Op);
  case CanonicalOp::V_CVT_F32_I32:
  case CanonicalOp::V_CVT_F32_U32:
  case CanonicalOp::V_CVT_F32_UBYTE0:
  case CanonicalOp::V_CVT_F32_UBYTE1:
  case CanonicalOp::V_CVT_F32_UBYTE2:
  case CanonicalOp::V_CVT_F32_UBYTE3: {
    if (Error Err = requireNoIntegerSourceModifiers(Ctx, Di, Op))
      return Err;
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseFloatConversion32(Ctx, Di, Op);
  }
  case CanonicalOp::V_CVT_I32_F32:
  case CanonicalOp::V_CVT_U32_F32:
  case CanonicalOp::V_CVT_F16_F32:
  case CanonicalOp::V_CVT_F32_F16: {
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseFloatConversion32(Ctx, Di, Op);
  }
  case CanonicalOp::V_FRACT_F32:
  case CanonicalOp::V_TRUNC_F32:
  case CanonicalOp::V_CEIL_F32:
  case CanonicalOp::V_RNDNE_F32:
  case CanonicalOp::V_FLOOR_F32:
  case CanonicalOp::V_EXP_F32:
  case CanonicalOp::V_LOG_F32:
  case CanonicalOp::V_RCP_F32:
  case CanonicalOp::V_RCP_IFLAG_F32:
  case CanonicalOp::V_TANH_F32:
  case CanonicalOp::V_S_EXP_F32:
  case CanonicalOp::V_S_LOG_F32:
  case CanonicalOp::V_S_RCP_F32:
  case CanonicalOp::V_S_RSQ_F32:
  case CanonicalOp::V_S_SQRT_F32:
  case CanonicalOp::V_RSQ_F32:
  case CanonicalOp::V_SQRT_F32:
  case CanonicalOp::V_SIN_F32:
  case CanonicalOp::V_COS_F32:
  case CanonicalOp::V_FREXP_EXP_I32_F32:
  case CanonicalOp::V_FREXP_MANT_F32: {
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseUnaryFloat32(Ctx, Di, Op);
  }
  case CanonicalOp::V_LDEXP_F32: {
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseLdexpFloat(Ctx, Di, Op);
  }
  case CanonicalOp::V_LDEXP_F64:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseLdexpFloat(Ctx, Di, Op);
  case CanonicalOp::V_CNDMASK_B32:
    if (Error Err = requireDefaultFloatOutputModifiers(Ctx, Di))
      return Err;
    return raiseCndMask32(Ctx, Di, Op);
  default:
    break;
  }

  // The 16-bit integer opcodes carry their register-half selection in the
  // source modifiers, so they are raised ahead of the checks that reject them.
  if (isInteger16Op(Di.CanonOp))
    return handleInteger16(Ctx, Di, Op);

  if (Error Err = requireNoIntegerSourceModifiers(Ctx, Di, Op))
    return Err;
  if (Error Err = requireNoOutputMultiplier(Ctx, Di))
    return Err;

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  switch (Di.CanonOp) {
  case CanonicalOp::V_MOV_B32:
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "integer move with clamp is not supported");
    return raiseMove32(Ctx, Di, Op);
  case CanonicalOp::V_MOV_B64:
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "integer move with clamp is not supported");
    return raiseMove64(Ctx, Di, Op);

  case CanonicalOp::V_NOT_B32:
  case CanonicalOp::V_BFREV_B32:
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32:
  case CanonicalOp::V_FFBH_I32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseUnaryBit32(Ctx, Di, Op);

  case CanonicalOp::V_AND_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateAnd(Src0, Src1, "and");
    });
  case CanonicalOp::V_OR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateOr(Src0, Src1, "or");
    });
  case CanonicalOp::V_XOR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateXor(Src0, Src1, "xor");
    });
  case CanonicalOp::V_XNOR_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Xor = B.CreateXor(Src0, Src1, "xnor_xor");
      return B.CreateNot(Xor, "xnor");
    });
  case CanonicalOp::V_BFM_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBitMask(Ctx, Op);
  case CanonicalOp::V_BCNT_U32_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBitCount(Ctx, Op);
  case CanonicalOp::V_MBCNT_LO_U32_B32:
    return raiseMaskedBitCountLow32(Ctx, Di, Op);
  case CanonicalOp::V_MBCNT_HI_U32_B32:
    return raiseMaskedBitCountHigh32(Ctx, Di, Op);
  case CanonicalOp::V_LSHLREV_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateShl(Src1, Amount, "lshl");
    });
  case CanonicalOp::V_LSHRREV_B32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateLShr(Src1, Amount, "lshr");
    });
  case CanonicalOp::V_ASHRREV_I32:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      Value *Amount = maskShiftAmount(B, Src0, 32);
      return B.CreateAShr(Src1, Amount, "ashr");
    });
  case CanonicalOp::V_LSHLREV_B64:
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    return raiseShiftLeft64(Ctx, Op);

  case CanonicalOp::V_LSHL_ADD_U32:
  case CanonicalOp::V_ADD_LSHL_U32:
  case CanonicalOp::V_LSHL_OR_B32:
  case CanonicalOp::V_AND_OR_B32:
  case CanonicalOp::V_OR3_B32:
  case CanonicalOp::V_XOR3_B32:
  case CanonicalOp::V_XAD_U32:
  case CanonicalOp::V_ALIGNBIT_B32:
  case CanonicalOp::V_BFE_U32:
  case CanonicalOp::V_BFE_I32:
  case CanonicalOp::V_BFI_B32:
  case CanonicalOp::V_PERM_B32: {
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();

    Value *Result;
    switch (Di.CanonOp) {
    case CanonicalOp::V_LSHL_ADD_U32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, Amount);
      Result = Ctx.B.CreateAdd(Shifted, Args->Src2, "lshl.add");
      break;
    }
    case CanonicalOp::V_ADD_LSHL_U32: {
      Value *Sum = Ctx.B.CreateAdd(Args->Src0, Args->Src1);
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Result = Ctx.B.CreateShl(Sum, Amount, "add.lshl");
      break;
    }
    case CanonicalOp::V_LSHL_OR_B32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Shifted = Ctx.B.CreateShl(Args->Src0, Amount);
      Result = Ctx.B.CreateOr(Shifted, Args->Src2, "lshl.or");
      break;
    }
    case CanonicalOp::V_AND_OR_B32: {
      Value *And = Ctx.B.CreateAnd(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateOr(And, Args->Src2, "and.or");
      break;
    }
    case CanonicalOp::V_OR3_B32: {
      Value *First = Ctx.B.CreateOr(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateOr(First, Args->Src2, "or3");
      break;
    }
    case CanonicalOp::V_XOR3_B32: {
      Value *First = Ctx.B.CreateXor(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateXor(First, Args->Src2, "xor3");
      break;
    }
    case CanonicalOp::V_XAD_U32: {
      Value *Xor = Ctx.B.CreateXor(Args->Src0, Args->Src1);
      Result = Ctx.B.CreateAdd(Xor, Args->Src2, "xad");
      break;
    }
    case CanonicalOp::V_ALIGNBIT_B32: {
      Value *Amount = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Result = Ctx.B.CreateIntrinsic(Intrinsic::fshr, {Ctx.B.getInt32Ty()},
                                     {Args->Src0, Args->Src1, Amount}, nullptr,
                                     "alignbit");
      break;
    }
    case CanonicalOp::V_BFE_U32:
    case CanonicalOp::V_BFE_I32: {
      Value *Offset = maskShiftAmount(Ctx.B, Args->Src1, 32);
      Value *Width = maskShiftAmount(Ctx.B, Args->Src2, 32);
      Value *WidthNonZero =
          Ctx.B.CreateICmpNE(Width, Ctx.B.getInt32(0), "bfe.width.nonzero");
      Value *SafeWidth = Ctx.B.CreateSelect(
          WidthNonZero, Width, Ctx.B.getInt32(1), "bfe.safe.width");
      Value *MaskHighBit = Ctx.B.CreateShl(Ctx.B.getInt32(1), SafeWidth);
      Value *Mask = Ctx.B.CreateSub(MaskHighBit, Ctx.B.getInt32(1), "bfe.mask");
      Value *Shifted = Di.CanonOp == CanonicalOp::V_BFE_I32
                           ? Ctx.B.CreateAShr(Args->Src0, Offset)
                           : Ctx.B.CreateLShr(Args->Src0, Offset);
      Value *Field = Ctx.B.CreateAnd(Shifted, Mask, "bfe.field");
      if (Di.CanonOp == CanonicalOp::V_BFE_I32) {
        Value *SignOffset = Ctx.B.CreateSub(SafeWidth, Ctx.B.getInt32(1));
        Value *SignBit = Ctx.B.CreateShl(Ctx.B.getInt32(1), SignOffset);
        Value *SignFlipped = Ctx.B.CreateXor(Field, SignBit);
        Field = Ctx.B.CreateSub(SignFlipped, SignBit, "bfe.sign.extend");
      }
      Result =
          Ctx.B.CreateSelect(WidthNonZero, Field, Ctx.B.getInt32(0), "bfe");
      break;
    }
    case CanonicalOp::V_BFI_B32: {
      Value *Selected = Ctx.B.CreateAnd(Args->Src0, Args->Src1);
      Value *InvertedMask = Ctx.B.CreateNot(Args->Src0);
      Value *Preserved = Ctx.B.CreateAnd(InvertedMask, Args->Src2);
      Result = Ctx.B.CreateOr(Selected, Preserved, "bfi");
      break;
    }
    case CanonicalOp::V_PERM_B32:
      Result = raiseBytePermute(Ctx.B, *Args);
      break;
    default:
      llvm_unreachable("not a ternary integer bit operation");
    }
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_BITOP3_B32: {
    if (*Clamp)
      return unsupportedInstruction(
          Ctx, Di, "integer bit operation does not define clamp");
    if (Op.nSrcs() < 4 || !Di.isImm(Op.srcIdx(3)))
      return unsupportedInstruction(
          Ctx, Di, "expected three data sources and an immediate truth table");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();

    Value *Result = emitBitOp3(Ctx.B, Args->Src0, Args->Src1, Args->Src2,
                               static_cast<uint8_t>(Op.srcImm(3)));
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_LSHRREV_B64:
  case CanonicalOp::V_ASHRREV_I64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit shift does not define clamp");
    if (Op.nSrcs() < 2)
      return unsupportedInstruction(Ctx, Di,
                                    "expected one destination and two sources");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Amount = Op.src(0);
    if (!Amount)
      return Amount.takeError();
    Expected<Value *> Source = Op.src64(1);
    if (!Source)
      return Source.takeError();
    Value *MaskedAmount = maskShiftAmount(Ctx.B, *Amount, 64);
    Value *Shift = Ctx.B.CreateZExt(MaskedAmount, Ctx.B.getInt64Ty());
    Value *Result = Di.CanonOp == CanonicalOp::V_LSHRREV_B64
                        ? Ctx.B.CreateLShr(*Source, Shift, "lshr64")
                        : Ctx.B.CreateAShr(*Source, Shift, "ashr64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_LSHL_ADD_U64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit shift-add does not define clamp");
    assert(Op.nSrcs() >= 3 && "shift-add instruction must have three sources");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src64(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    constexpr unsigned EncodedShiftValues = 8;
    constexpr unsigned MaxSupportedShift = 4;
    Value *EncodedShift = maskShiftAmount(Ctx.B, *S1, EncodedShiftValues);
    Value *ShiftSupported =
        Ctx.B.CreateICmpULE(EncodedShift, Ctx.B.getInt32(MaxSupportedShift));
    Value *Shift32 = Ctx.B.CreateSelect(ShiftSupported, EncodedShift,
                                        Ctx.B.getInt32(0), "shift.supported");
    Value *Shift = Ctx.B.CreateZExt(Shift32, Ctx.B.getInt64Ty());
    Value *Shifted = Ctx.B.CreateShl(*S0, Shift);
    Value *Result = Ctx.B.CreateAdd(Shifted, *S2, "lshl.add64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_NC_U32:
  case CanonicalOp::V_SUB_NC_U32:
  case CanonicalOp::V_SUBREV_NC_U32: {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    const bool IsAdd = Di.CanonOp == CanonicalOp::V_ADD_NC_U32;
    const bool Reverse = Di.CanonOp == CanonicalOp::V_SUBREV_NC_U32;
    Value *Lhs = Reverse ? *Src1 : *Src0;
    Value *Rhs = Reverse ? *Src0 : *Src1;
    Value *Result;
    if (*Clamp)
      Result = Ctx.B.CreateBinaryIntrinsic(
          IsAdd ? Intrinsic::uadd_sat : Intrinsic::usub_sat, Lhs, Rhs);
    else
      Result = IsAdd ? Ctx.B.CreateAdd(Lhs, Rhs, "add")
                     : Ctx.B.CreateSub(Lhs, Rhs, "sub");
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::uadd_with_overflow,
                             Intrinsic::uadd_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUB_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                             Intrinsic::usub_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUBREV_CO_U32:
    return handleBinaryCarry(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                             Intrinsic::usub_sat, /*Reverse=*/true, *Clamp);
  case CanonicalOp::V_ADD_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::uadd_with_overflow,
                               Intrinsic::uadd_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUB_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                               Intrinsic::usub_sat, /*Reverse=*/false, *Clamp);
  case CanonicalOp::V_SUBREV_CO_CI_U32:
    return handleBinaryCarryIn(Ctx, Di, Op, Intrinsic::usub_with_overflow,
                               Intrinsic::usub_sat, /*Reverse=*/true, *Clamp);

  case CanonicalOp::V_MIN_I32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_I32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::smax, Src0, Src1, {}, "max");
    });
  case CanonicalOp::V_MIN_U32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umin, Src0, Src1, {}, "min");
    });
  case CanonicalOp::V_MAX_U32:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateBinaryIntrinsic(Intrinsic::umax, Src0, Src1, {}, "max");
    });
  case CanonicalOp::V_MUL_U64:
    assert(!*Clamp && "integer arithmetic operation does not define clamp");
    return raiseBinary64(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
      return B.CreateMul(Src0, Src1, "mul64");
    });

  case CanonicalOp::V_MUL_I32_I24:
  case CanonicalOp::V_MUL_HI_I32_I24:
  case CanonicalOp::V_MUL_U32_U24:
  case CanonicalOp::V_MUL_HI_U32_U24: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MUL_I32_I24 ||
                        Di.CanonOp == CanonicalOp::V_MUL_HI_I32_I24;
    const bool High = Di.CanonOp == CanonicalOp::V_MUL_HI_I32_I24 ||
                      Di.CanonOp == CanonicalOp::V_MUL_HI_U32_U24;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    Value *A = extendLow24(Ctx.B, *Src0, Ctx.B.getInt64Ty(), Signed);
    Value *B = extendLow24(Ctx.B, *Src1, Ctx.B.getInt64Ty(), Signed);
    Value *Product = Ctx.B.CreateMul(A, B, "mul24.wide");
    Value *Result;
    if (High) {
      Value *Shifted = Signed ? Ctx.B.CreateAShr(Product, 32, "mul24.high")
                              : Ctx.B.CreateLShr(Product, 32, "mul24.high");
      Result = Ctx.B.CreateTrunc(Shifted, Ctx.B.getInt32Ty(), "mul24.hi");
    } else if (*Clamp) {
      if (Signed) {
        Value *Lo = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MIN);
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MAX);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Product, Lo);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Product, Hi);
      } else {
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), UINT32_MAX);
        Product = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umin, Product, Hi);
      }
      Result = Ctx.B.CreateTrunc(Product, Ctx.B.getInt32Ty(), "mul24.clamp");
    } else {
      Result = Ctx.B.CreateTrunc(Product, Ctx.B.getInt32Ty(), "mul24");
    }
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_NC_U64:
  case CanonicalOp::V_SUB_NC_U64:
  case CanonicalOp::V_ADD_I32:
  case CanonicalOp::V_SUB_I32: {
    const bool Is64 = Di.CanonOp == CanonicalOp::V_ADD_NC_U64 ||
                      Di.CanonOp == CanonicalOp::V_SUB_NC_U64;
    const bool Signed = !Is64;
    const bool IsAdd = Di.CanonOp == CanonicalOp::V_ADD_NC_U64 ||
                       Di.CanonOp == CanonicalOp::V_ADD_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0, Is64);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1, Is64);
    if (!Src1)
      return Src1.takeError();
    Intrinsic::ID Saturating =
        Signed ? (IsAdd ? Intrinsic::sadd_sat : Intrinsic::ssub_sat)
               : (IsAdd ? Intrinsic::uadd_sat : Intrinsic::usub_sat);
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Saturating, *Src0, *Src1)
               : Ctx.B.CreateBinOp(IsAdd ? Instruction::Add : Instruction::Sub,
                                   *Src0, *Src1);
    if (Is64)
      Ctx.registers().writeReg64(*Dst, Result);
    else
      Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MUL_LO_U32:
    // Clamp is reserved for this opcode. Refuse malformed input rather than
    // asserting while decoding an external binary.
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di, "multiply does not define clamp");
    return handleBinary(Ctx, Op, Instruction::Mul, false);
  case CanonicalOp::V_MUL_HI_U32:
  case CanonicalOp::V_MUL_HI_I32: {
    // Clamp is reserved for these opcodes as well.
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di, "multiply does not define clamp");
    const bool Signed = Di.CanonOp == CanonicalOp::V_MUL_HI_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src0 = Op.src(0);
    if (!Src0)
      return Src0.takeError();
    Expected<Value *> Src1 = Op.src(1);
    if (!Src1)
      return Src1.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*Src0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*Src0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*Src1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*Src1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mul.wide");
    Value *High = Signed ? Ctx.B.CreateAShr(Product, 32, "mul.high")
                         : Ctx.B.CreateLShr(Product, 32, "mul.high");
    Ctx.registers().writeReg32(
        *Dst, Ctx.B.CreateTrunc(High, Ctx.B.getInt32Ty(), "mul.hi"));
    return Error::success();
  }

  case CanonicalOp::V_MAD_I32_I24:
  case CanonicalOp::V_MAD_U32_U24: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_I32_I24;
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *Result;
    if (*Clamp) {
      Value *A = extendLow24(Ctx.B, Args->Src0, Ctx.B.getInt64Ty(), Signed);
      Value *B = extendLow24(Ctx.B, Args->Src1, Ctx.B.getInt64Ty(), Signed);
      Value *C = Signed ? Ctx.B.CreateSExt(Args->Src2, Ctx.B.getInt64Ty())
                        : Ctx.B.CreateZExt(Args->Src2, Ctx.B.getInt64Ty());
      Value *Product = Ctx.B.CreateMul(A, B);
      Value *Wide = Ctx.B.CreateAdd(Product, C, "mad.wide");
      if (Signed) {
        Value *Lo = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MIN);
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), INT32_MAX);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Wide, Lo);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Wide, Hi);
      } else {
        Value *Hi = ConstantInt::get(Ctx.B.getInt64Ty(), UINT32_MAX);
        Wide = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umin, Wide, Hi);
      }
      Result = Ctx.B.CreateTrunc(Wide, Ctx.B.getInt32Ty(), "mad.clamp");
    } else {
      Value *A = extendLow24(Ctx.B, Args->Src0, Ctx.B.getInt32Ty(), Signed);
      Value *B = extendLow24(Ctx.B, Args->Src1, Ctx.B.getInt32Ty(), Signed);
      // LLVM has no integer FMA intrinsic; separate integer operations retain
      // the instruction's wrapping multiply-add semantics.
      Value *Product = Ctx.B.CreateMul(A, B);
      Result = Ctx.B.CreateAdd(Product, Args->Src2, "mad24");
    }
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MAD_U32: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "instruction does not define clamp");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src(2);
    if (!S2)
      return S2.takeError();
    Value *Product = Ctx.B.CreateMul(*S0, *S1, "mad.mul");
    Ctx.registers().writeReg32(*Dst, Ctx.B.CreateAdd(Product, *S2, "mad"));
    return Error::success();
  }

  case CanonicalOp::V_ADD3_U32: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "instruction does not define clamp");
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *First = Ctx.B.CreateAdd(Args->Src0, Args->Src1, "add3.first");
    Value *Result = Ctx.B.CreateAdd(First, Args->Src2, "add3");
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_ADD_MIN_I32:
  case CanonicalOp::V_ADD_MAX_I32:
  case CanonicalOp::V_ADD_MIN_U32:
  case CanonicalOp::V_ADD_MAX_U32: {
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    const bool Signed = Di.CanonOp == CanonicalOp::V_ADD_MIN_I32 ||
                        Di.CanonOp == CanonicalOp::V_ADD_MAX_I32;
    const bool Max = Di.CanonOp == CanonicalOp::V_ADD_MAX_I32 ||
                     Di.CanonOp == CanonicalOp::V_ADD_MAX_U32;
    Value *Sum = Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Args->Src0, Args->Src1);
    Intrinsic::ID MinMax = Signed ? (Max ? Intrinsic::smax : Intrinsic::smin)
                                  : (Max ? Intrinsic::umax : Intrinsic::umin);
    Value *Result = Ctx.B.CreateBinaryIntrinsic(MinMax, Sum, Args->Src2);
    if (*Clamp && Signed)
      Result = Ctx.B.CreateBinaryIntrinsic(
          Intrinsic::smax, Result, ConstantInt::get(Ctx.B.getInt32Ty(), 0));
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MIN3_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smin, Intrinsic::smin,
                               "min3");
  case CanonicalOp::V_MAX3_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smax, Intrinsic::smax,
                               "max3");
  case CanonicalOp::V_MIN3_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umin, Intrinsic::umin,
                               "min3");
  case CanonicalOp::V_MAX3_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umax, Intrinsic::umax,
                               "max3");
  case CanonicalOp::V_MED3_I32: {
    Expected<TernaryOperands> Args = Op.readTernary32();
    if (!Args)
      return Args.takeError();
    Value *Lo =
        Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Args->Src0, Args->Src1);
    Value *Hi =
        Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Args->Src0, Args->Src1);
    Value *Upper = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Hi, Args->Src2);
    Value *Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Lo, Upper);
    Ctx.registers().writeReg32(Args->Dst, Result);
    return Error::success();
  }
  case CanonicalOp::V_MINMAX_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smin, Intrinsic::smax,
                               "minmax");
  case CanonicalOp::V_MAXMIN_I32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::smax, Intrinsic::smin,
                               "maxmin");
  case CanonicalOp::V_MINMAX_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umin, Intrinsic::umax,
                               "minmax");
  case CanonicalOp::V_MAXMIN_U32:
    return handleTernaryMinMax(Ctx, Op, Intrinsic::umax, Intrinsic::umin,
                               "maxmin");

  case CanonicalOp::V_MIN_I64:
  case CanonicalOp::V_MAX_I64:
  case CanonicalOp::V_MIN_U64:
  case CanonicalOp::V_MAX_U64: {
    if (*Clamp)
      return unsupportedInstruction(Ctx, Di,
                                    "64-bit min/max does not define clamp");
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src64(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src64(1);
    if (!S1)
      return S1.takeError();
    Intrinsic::ID ID;
    switch (Di.CanonOp) {
    case CanonicalOp::V_MIN_I64:
      ID = Intrinsic::smin;
      break;
    case CanonicalOp::V_MAX_I64:
      ID = Intrinsic::smax;
      break;
    case CanonicalOp::V_MIN_U64:
      ID = Intrinsic::umin;
      break;
    case CanonicalOp::V_MAX_U64:
      ID = Intrinsic::umax;
      break;
    default:
      llvm_unreachable("not a 64-bit min/max");
    }
    Ctx.registers().writeReg64(*Dst, Ctx.B.CreateBinaryIntrinsic(ID, *S0, *S1));
    return Error::success();
  }

  case CanonicalOp::V_MAD_NC_U64_U32:
  case CanonicalOp::V_MAD_NC_I64_I32: {
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_NC_I64_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*S0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*S1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mad64.mul");
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Product, *S2)
               : Ctx.B.CreateAdd(Product, *S2, "mad64");
    Ctx.registers().writeReg64(*Dst, Result);
    return Error::success();
  }

  case CanonicalOp::V_MAD_U64_U32:
  case CanonicalOp::V_MAD_I64_I32: {
    if (Di.NumDefs != 2 || Op.nSrcs() < 3)
      return unsupportedInstruction(
          Ctx, Di, "expected two destinations and three sources");
    const bool Signed = Di.CanonOp == CanonicalOp::V_MAD_I64_I32;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> S0 = Op.src(0);
    if (!S0)
      return S0.takeError();
    Expected<Value *> S1 = Op.src(1);
    if (!S1)
      return S1.takeError();
    Expected<Value *> S2 = Op.src64(2);
    if (!S2)
      return S2.takeError();
    Value *A = Signed ? Ctx.B.CreateSExt(*S0, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S0, Ctx.B.getInt64Ty());
    Value *B = Signed ? Ctx.B.CreateSExt(*S1, Ctx.B.getInt64Ty())
                      : Ctx.B.CreateZExt(*S1, Ctx.B.getInt64Ty());
    Value *Product = Ctx.B.CreateMul(A, B, "mad64.mul");
    Value *Pair = Ctx.B.CreateIntrinsic(Intrinsic::uadd_with_overflow,
                                        {Ctx.B.getInt64Ty()}, {Product, *S2});
    Value *Wrapped = Ctx.B.CreateExtractValue(Pair, 0, "mad64");
    Value *Carry = Ctx.B.CreateExtractValue(Pair, 1, "mad64.carry");
    if (Signed) {
      Value *ProductSign =
          Ctx.B.CreateICmpSLT(Product, ConstantInt::get(Ctx.B.getInt64Ty(), 0));
      Value *AccumulatorSign =
          Ctx.B.CreateICmpSLT(*S2, ConstantInt::get(Ctx.B.getInt64Ty(), 0));
      Value *SignsDiffer = Ctx.B.CreateXor(ProductSign, AccumulatorSign);
      Carry = Ctx.B.CreateXor(SignsDiffer, Carry, "mad64.high.bit");
    }
    Value *Result =
        *Clamp ? Ctx.B.CreateBinaryIntrinsic(Signed ? Intrinsic::sadd_sat
                                                    : Intrinsic::uadd_sat,
                                             Product, *S2)
               : Wrapped;
    Ctx.registers().writeReg64(*Dst, Result);
    return writeCarryOut(Ctx, Di, Op, Carry);
  }

  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::transpiler
