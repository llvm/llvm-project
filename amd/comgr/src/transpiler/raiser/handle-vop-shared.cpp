//===- handle-vop-shared.cpp - Shared VOP lowering helpers ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handle-vop-shared.h"

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"

#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include <cassert>

#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

namespace COMGR::transpiler {

Error raiseMove32(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();
  Ctx.registers().writeReg32(*Dst, *Src);
  return Error::success();
}

Error raiseMove64(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src64(0);
  if (!Src)
    return Src.takeError();
  Ctx.registers().writeReg64(*Dst, *Src);
  return Error::success();
}

Error raiseUnaryBit32(RaiseContext &Ctx, const DecodedInst &Di,
                      OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() < 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src = Op.src(0);
  if (!Src)
    return Src.takeError();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_NOT_B32:
    Result = Ctx.B.CreateNot(*Src, "not");
    break;
  case CanonicalOp::V_BFREV_B32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::bitreverse, *Src, nullptr,
                                        "bfrev");
    break;
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32: {
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_FFBH_U32 ? Intrinsic::ctlz
                                                             : Intrinsic::cttz;
    Value *Count = Ctx.B.CreateIntrinsic(
        ID, {Ctx.B.getInt32Ty()}, {*Src, Ctx.B.getFalse()}, nullptr, "ffb");
    Value *IsZero = Ctx.B.CreateICmpEQ(*Src, Ctx.B.getInt32(0), "ffb.zero");
    Result = Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Count, "ffb");
    break;
  }
  case CanonicalOp::V_FFBH_I32: {
    Value *Sign = Ctx.B.CreateAShr(*Src, Ctx.B.getInt32(31), "ffbh.sign");
    Value *Normalized = Ctx.B.CreateXor(*Src, Sign, "ffbh.normalized");
    Value *Count = Ctx.B.CreateIntrinsic(Intrinsic::ctlz, {Ctx.B.getInt32Ty()},
                                         {Normalized, Ctx.B.getFalse()},
                                         nullptr, "ffbh.count");
    Value *AllSignBits =
        Ctx.B.CreateICmpEQ(Normalized, Ctx.B.getInt32(0), "ffbh.uniform");
    Result =
        Ctx.B.CreateSelect(AllSignBits, Ctx.B.getInt32(-1), Count, "ffbh.i32");
    break;
  }
  default:
    llvm_unreachable("not a unary integer bit operation");
  }
  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseUnaryFloat32(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Source = Op.srcF32(0);
  if (!Source)
    return Source.takeError();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_FRACT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_fract, *Source,
                                        nullptr, "fract");
    break;
  case CanonicalOp::V_TRUNC_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::trunc, *Source, nullptr, "trunc");
    break;
  case CanonicalOp::V_CEIL_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::ceil, *Source, nullptr, "ceil");
    break;
  case CanonicalOp::V_RNDNE_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::roundeven, *Source, nullptr,
                                        "rndne");
    break;
  case CanonicalOp::V_FLOOR_F32:
    Result =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::floor, *Source, nullptr, "floor");
    break;
  // These intrinsics directly model the source VALU operations, including
  // their approximate results and denormal behavior. V_SIN_F32 and V_COS_F32
  // also interpret their inputs as fractions of 2*pi. Generic LLVM math
  // intrinsics have libm semantics and may require refinement sequences.
  case CanonicalOp::V_EXP_F32:
  case CanonicalOp::V_S_EXP_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_exp2, *Source,
                                        nullptr, "exp");
    break;
  case CanonicalOp::V_LOG_F32:
  case CanonicalOp::V_S_LOG_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_log, *Source, nullptr,
                                        "log");
    break;
  case CanonicalOp::V_RCP_F32:
  case CanonicalOp::V_S_RCP_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_rcp, *Source, nullptr,
                                        "rcp");
    break;
  case CanonicalOp::V_RCP_IFLAG_F32:
    Result = Ctx.B.CreateCall(
        InlineAsm::get(
            FunctionType::get(Ctx.B.getFloatTy(), Ctx.B.getFloatTy(), false),
            "v_rcp_iflag_f32 $0, $1", "=v,v", true),
        {*Source}, "rcp.iflag");
    break;
  case CanonicalOp::V_RSQ_F32:
  case CanonicalOp::V_S_RSQ_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_rsq, *Source, nullptr,
                                        "rsq");
    break;
  case CanonicalOp::V_SQRT_F32:
  case CanonicalOp::V_S_SQRT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_sqrt, *Source,
                                        nullptr, "sqrt");
    break;
  case CanonicalOp::V_SIN_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_sin, *Source, nullptr,
                                        "sin");
    break;
  case CanonicalOp::V_COS_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_cos, *Source, nullptr,
                                        "cos");
    break;
  case CanonicalOp::V_TANH_F32:
    if (!Ctx.Projection.TargetSTI.hasFeature(AMDGPU::FeatureTanhInsts))
      return unsupportedInstruction(Ctx, Di,
                                    "target does not support v_tanh_f32");
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_tanh, *Source,
                                        nullptr, "tanh");
    break;
  case CanonicalOp::V_FREXP_EXP_I32_F32:
    Result = Ctx.B.CreateIntrinsic(Intrinsic::amdgcn_frexp_exp,
                                   {Ctx.B.getInt32Ty(), Ctx.B.getFloatTy()},
                                   {*Source}, nullptr, "frexp.exp");
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  case CanonicalOp::V_FREXP_MANT_F32:
    Result = Ctx.B.CreateUnaryIntrinsic(Intrinsic::amdgcn_frexp_mant, *Source,
                                        nullptr, "frexp.mant");
    break;
  default:
    llvm_unreachable("not a unary F32 operation");
  }

  Value *ResultBits = Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty());
  Ctx.registers().writeReg32(*Dst, ResultBits);
  return Error::success();
}

Error raiseUnaryFloat64(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getDoubleTy()))
    return Err;
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Source = Op.srcF64(0);
  if (!Source)
    return Source.takeError();

  Intrinsic::ID ID;
  switch (Di.CanonOp) {
  case CanonicalOp::V_TRUNC_F64:
    ID = Intrinsic::trunc;
    break;
  case CanonicalOp::V_CEIL_F64:
    ID = Intrinsic::ceil;
    break;
  case CanonicalOp::V_RNDNE_F64:
    ID = Intrinsic::roundeven;
    break;
  case CanonicalOp::V_FLOOR_F64:
    ID = Intrinsic::floor;
    break;
  case CanonicalOp::V_RCP_F64:
    ID = Intrinsic::amdgcn_rcp;
    break;
  case CanonicalOp::V_RSQ_F64:
    ID = Intrinsic::amdgcn_rsq;
    break;
  default:
    llvm_unreachable("not a unary F64 operation");
  }
  Value *Result = Ctx.B.CreateUnaryIntrinsic(ID, *Source);
  Ctx.registers().writeReg64(*Dst,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt64Ty()));
  return Error::success();
}

/// Flush a subnormal operand or result according to its source FP mode.
static Value *flushDenormal(IRBuilder<> &B, Value *V, bool IsOutput) {
  Type *Ty = V->getType();
  const fltSemantics &Semantics = Ty->getFltSemantics();
  DenormalMode Mode =
      B.GetInsertBlock()->getParent()->getDenormalMode(Semantics);
  DenormalMode::DenormalModeKind Kind = IsOutput ? Mode.Output : Mode.Input;
  if (Kind == DenormalMode::IEEE)
    return V;
  assert(Kind == DenormalMode::PreserveSign &&
         "unexpected source denormal mode");
  Type *BitsTy = B.getIntNTy(Ty->getPrimitiveSizeInBits());
  Value *Bits = B.CreateBitCast(V, BitsTy);
  APInt SignMask = APInt::getSignMask(Ty->getPrimitiveSizeInBits());
  Value *Magnitude = B.CreateAnd(Bits, ~SignMask);
  Value *IsTiny = B.CreateICmpULT(
      Magnitude,
      ConstantInt::get(
          BitsTy, APFloat::getSmallestNormalized(Semantics).bitcastToAPInt()));
  Value *Sign = B.CreateAnd(Bits, SignMask);
  return B.CreateBitCast(B.CreateSelect(IsTiny, Sign, Bits), Ty);
}

/// Quiet result NaNs and enforce the source output denormal mode.
static Value *getFloatResultBits(IRBuilder<> &B, Value *Result) {
  Type *Ty = Result->getType();
  unsigned Width = Ty->getPrimitiveSizeInBits();
  Type *BitsTy = B.getIntNTy(Width);
  Value *Bits = B.CreateBitCast(flushDenormal(B, Result, true), BitsTy);
  APInt Infinity = APFloat::getInf(Ty->getFltSemantics()).bitcastToAPInt();
  APInt QuietBit =
      APFloat::getQNaN(Ty->getFltSemantics()).bitcastToAPInt() ^ Infinity;
  Value *Magnitude = B.CreateAnd(Bits, APInt::getSignedMaxValue(Width));
  Value *IsNaN = B.CreateICmpUGT(Magnitude, ConstantInt::get(BitsTy, Infinity));
  // Ordinary FP operations may propagate a signaling NaN unchanged.
  return B.CreateSelect(IsNaN, B.CreateOr(Bits, QuietBit), Bits);
}

Error raiseFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op) {
  assert(Di.NumDefs == 1 && Op.nSrcs() == 2 &&
         "expected one destination and two sources");

  bool Is64 = Di.CanonOp == CanonicalOp::V_MUL_F64 ||
              Di.CanonOp == CanonicalOp::V_ADD_F64;
  Type *Ty = Is64 ? Ctx.B.getDoubleTy() : Ctx.B.getFloatTy();
  if (Error Err = Ctx.validateFPEnvironment(Di, Ty)) {
    return Err;
  }

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<Value *> Left = Is64 ? Op.srcF64(0) : Op.srcF32(0);
  if (!Left)
    return Left.takeError();
  Expected<Value *> Right = Is64 ? Op.srcF64(1) : Op.srcF32(1);
  if (!Right)
    return Right.takeError();

  Instruction::BinaryOps Opcode;
  switch (Di.CanonOp) {
  case CanonicalOp::V_ADD_F32:
  case CanonicalOp::V_ADD_F64:
    Opcode = Instruction::FAdd;
    break;
  case CanonicalOp::V_MUL_F32:
  case CanonicalOp::V_MUL_F64:
    Opcode = Instruction::FMul;
    break;
  case CanonicalOp::V_SUB_F32:
  case CanonicalOp::V_SUBREV_F32:
    Opcode = Instruction::FSub;
    break;
  default:
    llvm_unreachable("not binary floating-point arithmetic");
  }
  bool ReverseOperands = Di.CanonOp == CanonicalOp::V_SUBREV_F32;
  Value *Lhs = flushDenormal(Ctx.B, ReverseOperands ? *Right : *Left, false);
  Value *Rhs = flushDenormal(Ctx.B, ReverseOperands ? *Left : *Right, false);
  Value *Bits = getFloatResultBits(Ctx.B, Ctx.B.CreateBinOp(Opcode, Lhs, Rhs));
  if (Is64)
    Ctx.registers().writeReg64(*Destination, Bits);
  else
    Ctx.registers().writeReg32(*Destination, Bits);
  return Error::success();
}

Error raiseNumericExtremum32(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op) {
  assert(Di.NumDefs == 1 && Op.nSrcs() == 2 &&
         "expected one destination and two sources");
  assert((Di.CanonOp == CanonicalOp::V_MAX_NUM_F32 ||
          Di.CanonOp == CanonicalOp::V_MIN_NUM_F32) &&
         "expected numeric minimum or maximum");
  // Legacy min/max share these canonical opcodes but have different NaN rules.
  if (Ctx.Projection.SourceSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode))
    return unsupportedInstruction(
        Ctx, Di, "legacy floating-point min/max semantics are not supported");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;
  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<Value *> Left = Op.srcF32(0);
  if (!Left)
    return Left.takeError();
  Expected<Value *> Right = Op.srcF32(1);
  if (!Right)
    return Right.takeError();
  Value *LeftBits = Ctx.B.CreateBitCast(flushDenormal(Ctx.B, *Left, false),
                                        Ctx.B.getInt32Ty());
  Value *RightBits = Ctx.B.CreateBitCast(flushDenormal(Ctx.B, *Right, false),
                                         Ctx.B.getInt32Ty());
  auto IsNaN = [&](Value *Bits) {
    return Ctx.B.CreateICmpUGT(Ctx.B.CreateAnd(Bits, 0x7fffffff),
                               Ctx.B.getInt32(0x7f800000));
  };
  // Signed integer order after reflecting negative encodings also orders -0
  // before +0, without flushing a selected subnormal result.
  auto Ordered = [&](Value *Bits) {
    Value *Negative = Ctx.B.CreateICmpSLT(Bits, Ctx.B.getInt32(0));
    return Ctx.B.CreateXor(Bits, Ctx.B.CreateSelect(Negative,
                                                    Ctx.B.getInt32(0x7fffffff),
                                                    Ctx.B.getInt32(0)));
  };
  Value *ChooseLeft =
      Di.CanonOp == CanonicalOp::V_MAX_NUM_F32
          ? Ctx.B.CreateICmpSGT(Ordered(LeftBits), Ordered(RightBits))
          : Ctx.B.CreateICmpSLT(Ordered(LeftBits), Ordered(RightBits));
  Value *Bits = Ctx.B.CreateSelect(ChooseLeft, LeftBits, RightBits);
  Value *LeftNaN = IsNaN(LeftBits);
  Value *RightNaN = IsNaN(RightBits);
  Bits = Ctx.B.CreateSelect(LeftNaN, RightBits, Bits);
  Bits = Ctx.B.CreateSelect(RightNaN, LeftBits, Bits);
  Value *BothNaN = Ctx.B.CreateAnd(LeftNaN, RightNaN);
  Bits =
      Ctx.B.CreateSelect(BothNaN, Ctx.B.CreateOr(LeftBits, 0x00400000), Bits);
  Ctx.registers().writeReg32(*Destination, Bits);
  return Error::success();
}

Error raiseFloatMac(RaiseContext &Ctx, const DecodedInst &Di,
                    OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 3)
    return unsupportedInstruction(
        Ctx, Di, "expected floating-point multiply-add operands");
  bool Is64 = Di.CanonOp == CanonicalOp::V_FMAC_F64 ||
              Di.CanonOp == CanonicalOp::V_FMAMK_F64 ||
              Di.CanonOp == CanonicalOp::V_FMA_F64;
  Type *Ty = Is64 ? Ctx.B.getDoubleTy() : Ctx.B.getFloatTy();
  if (Error Err = Ctx.validateFPEnvironment(Di, Ty))
    return Err;
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  auto Read = [&](unsigned I) { return Is64 ? Op.srcF64(I) : Op.srcF32(I); };
  Expected<Value *> Src0 = Read(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Read(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> Src2 = Read(2);
  if (!Src2)
    return Src2.takeError();
  Value *Result = Ctx.B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                        {flushDenormal(Ctx.B, *Src0, false),
                                         flushDenormal(Ctx.B, *Src1, false),
                                         flushDenormal(Ctx.B, *Src2, false)});
  Value *Bits = getFloatResultBits(Ctx.B, Result);
  if (Is64)
    Ctx.registers().writeReg64(*Dst, Bits);
  else
    Ctx.registers().writeReg32(*Dst, Bits);
  return Error::success();
}

Error raiseFloatConversion32(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;
  if (Di.CanonOp == CanonicalOp::V_CVT_F16_F32) {
    if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getHalfTy()))
      return Err;
  }

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_CVT_F32_I32:
  case CanonicalOp::V_CVT_F32_U32: {
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    Value *Converted = Di.CanonOp == CanonicalOp::V_CVT_F32_I32
                           ? Ctx.B.CreateSIToFP(*Source, Ctx.B.getFloatTy())
                           : Ctx.B.CreateUIToFP(*Source, Ctx.B.getFloatTy());
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  case CanonicalOp::V_CVT_I32_F32:
  case CanonicalOp::V_CVT_U32_F32: {
    Expected<Value *> Source = Op.srcF32(0);
    if (!Source)
      return Source.takeError();
    Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_CVT_I32_F32
                           ? Intrinsic::fptosi_sat
                           : Intrinsic::fptoui_sat;
    Result = Ctx.B.CreateIntrinsic(ID, {Ctx.B.getInt32Ty(), Ctx.B.getFloatTy()},
                                   {*Source});
    break;
  }
  case CanonicalOp::V_CVT_F16_F32: {
    Expected<Value *> Source = Op.srcF32(0);
    if (!Source)
      return Source.takeError();
    Value *Half = Ctx.B.CreateFPTrunc(flushDenormal(Ctx.B, *Source, false),
                                      Ctx.B.getHalfTy(), "cvt");
    return writeDestination16(Ctx, Op, getFloatResultBits(Ctx.B, Half));
  }
  case CanonicalOp::V_CVT_F32_F16: {
    Expected<Value *> Source = Op.srcF16(0);
    if (!Source)
      return Source.takeError();
    Value *Converted = Ctx.B.CreateFPExt(*Source, Ctx.B.getFloatTy(), "cvt");
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  case CanonicalOp::V_CVT_F32_UBYTE0:
  case CanonicalOp::V_CVT_F32_UBYTE1:
  case CanonicalOp::V_CVT_F32_UBYTE2:
  case CanonicalOp::V_CVT_F32_UBYTE3: {
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    unsigned ByteIndex = Di.CanonOp == CanonicalOp::V_CVT_F32_UBYTE0   ? 0
                         : Di.CanonOp == CanonicalOp::V_CVT_F32_UBYTE1 ? 1
                         : Di.CanonOp == CanonicalOp::V_CVT_F32_UBYTE2 ? 2
                                                                       : 3;
    Value *Byte = *Source;
    if (ByteIndex != 0)
      Byte = Ctx.B.CreateLShr(Byte, ByteIndex * 8, "cvt.byte.shift");
    Byte = Ctx.B.CreateAnd(Byte, Ctx.B.getInt32(0xff), "cvt.byte");
    Value *Converted = Ctx.B.CreateUIToFP(Byte, Ctx.B.getFloatTy());
    Result = Ctx.B.CreateBitCast(Converted, Ctx.B.getInt32Ty());
    break;
  }
  default:
    llvm_unreachable("not a 32-bit floating-point conversion");
  }

  Ctx.registers().writeReg32(*Dst, Result);
  return Error::success();
}

Error raiseFloatConversion64(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op) {
  if (Di.NumDefs != 1 || Op.nSrcs() != 1)
    return unsupportedInstruction(Ctx, Di,
                                  "expected one destination and one source");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getDoubleTy()))
    return Err;
  if (Di.CanonOp == CanonicalOp::V_CVT_F32_F64 ||
      Di.CanonOp == CanonicalOp::V_CVT_F64_F32)
    if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
      return Err;
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  if (Di.CanonOp == CanonicalOp::V_CVT_F64_F32) {
    Expected<Value *> Source = Op.srcF32(0);
    if (!Source)
      return Source.takeError();
    Value *Result = Ctx.B.CreateFPExt(*Source, Ctx.B.getDoubleTy());
    Ctx.registers().writeReg64(*Dst,
                               Ctx.B.CreateBitCast(Result, Ctx.B.getInt64Ty()));
  } else if (Di.CanonOp == CanonicalOp::V_CVT_F64_I32 ||
             Di.CanonOp == CanonicalOp::V_CVT_F64_U32) {
    if (Op.srcMod(0) != 0)
      return unsupportedInstruction(
          Ctx, Di, "integer source modifiers are not supported");
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    Value *Result = Di.CanonOp == CanonicalOp::V_CVT_F64_I32
                        ? Ctx.B.CreateSIToFP(*Source, Ctx.B.getDoubleTy())
                        : Ctx.B.CreateUIToFP(*Source, Ctx.B.getDoubleTy());
    Ctx.registers().writeReg64(*Dst,
                               Ctx.B.CreateBitCast(Result, Ctx.B.getInt64Ty()));
  } else {
    if (Op.srcMod(0) & ~(SISrcMods::NEG | SISrcMods::ABS))
      return unsupportedInstruction(Ctx, Di, "unsupported f64 source modifier");
    Expected<Value *> Source = Op.src64(0);
    if (!Source)
      return Source.takeError();
    const MCOperand &Operand = Di.Inst.getOperand(Op.srcIdx(0));
    if (Operand.isExpr()) {
      const auto *Expr = dyn_cast<AMDGPUMCExpr>(Operand.getExpr());
      if (Expr && Expr->getKind() == AMDGPUMCExpr::AGVK_Lit) {
        // A 32-bit FP64 literal supplies the high word of the value.
        std::optional<int64_t> Literal =
            evalOperandAsConst(Di.Inst, Op.srcIdx(0));
        if (!Literal)
          return unsupportedInstruction(Ctx, Di, "unresolved FP64 literal");
        uint64_t Bits = static_cast<uint64_t>(static_cast<uint32_t>(*Literal))
                        << 32;
        *Source = Ctx.B.getInt64(Bits);
      }
    }
    Value *Float = Ctx.B.CreateBitCast(*Source, Ctx.B.getDoubleTy());
    Float = Op.applyMods(0, Float);
    Value *Result;
    if (Di.CanonOp == CanonicalOp::V_CVT_F32_F64) {
      Value *Truncated = Ctx.B.CreateFPTrunc(Float, Ctx.B.getFloatTy());
      Result = Ctx.B.CreateBitCast(Truncated, Ctx.B.getInt32Ty());
    } else {
      Intrinsic::ID ID = Di.CanonOp == CanonicalOp::V_CVT_I32_F64
                             ? Intrinsic::fptosi_sat
                             : Intrinsic::fptoui_sat;
      Result = Ctx.B.CreateIntrinsic(
          ID, {Ctx.B.getInt32Ty(), Ctx.B.getDoubleTy()}, {Float});
    }
    Ctx.registers().writeReg32(*Dst, Result);
  }
  return Error::success();
}

Expected<Value *> readCndMaskCondition(RaiseContext &Ctx, OperandResolver &Op) {
  if (Op.nSrcs() == 2)
    return Ctx.registers().regFile().loadVCC(Ctx.B);

  Expected<Value *> KnownCondition = Op.srcWaveMaskI1(2);
  if (!KnownCondition)
    return KnownCondition.takeError();
  if (*KnownCondition)
    return *KnownCondition;

  // Use the full SGPR mask when no per-lane condition is available.
  Expected<Value *> Mask = Op.srcExecWidth(2);
  if (!Mask)
    return Mask.takeError();
  return Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, *Mask);
}

Error raiseCndMask32(RaiseContext &Ctx, const DecodedInst &Di,
                     OperandResolver &Op) {
  if (Di.NumDefs != 1 || (Op.nSrcs() != 2 && Op.nSrcs() != 3))
    return unsupportedInstruction(
        Ctx, Di, "expected one destination and two values plus a condition");

  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Source0 = Op.applyMods(0, Args->Src0);
  Value *Source1 = Op.applyMods(1, Args->Src1);

  Expected<Value *> Condition = readCndMaskCondition(Ctx, Op);
  if (!Condition)
    return Condition.takeError();

  Value *Result = Ctx.B.CreateSelect(*Condition, Source1, Source0, "cndmask");
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

Error raiseBinary32(RaiseContext &Ctx, OperandResolver &Op,
                    BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary32();
  if (!Args)
    return Args.takeError();
  Value *Result = Build(Ctx.B, Args->Src0, Args->Src1);
  Ctx.registers().writeReg32(Args->Dst, Result);
  return Error::success();
}

Error raiseBinary64(RaiseContext &Ctx, OperandResolver &Op,
                    BinaryBuilder Build) {
  Expected<BinaryOperands> Args = Op.readBinary64();
  if (!Args)
    return Args.takeError();
  Value *Result = Build(Ctx.B, Args->Src0, Args->Src1);
  Ctx.registers().writeReg64(Args->Dst, Result);
  return Error::success();
}

Expected<bool> readClamp(RaiseContext &Ctx, const DecodedInst &Di) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                    AMDGPU::OpName::clamp);
  if (Index < 0)
    return false;
  if (!Di.isImm(Index))
    return unsupportedInstruction(Ctx, Di, "clamp operand is not immediate");
  int64_t Clamp = Di.getImm(Index);
  if (Clamp != 0 && Clamp != 1)
    return unsupportedInstruction(Ctx, Di, "clamp operand is not 0 or 1");
  return Clamp != 0;
}

Value *emitBitOp3(IRBuilder<> &B, Value *Src0, Value *Src1, Value *Src2,
                  uint8_t TruthTable) {
  Value *Not0 = B.CreateNot(Src0);
  Value *Not1 = B.CreateNot(Src1);
  Value *Not2 = B.CreateNot(Src2);
  constexpr unsigned TruthTableSize = 8;
  Value *Result = nullptr;
  for (unsigned Index = 0; Index != TruthTableSize; ++Index) {
    if (!(TruthTable & (1u << Index)))
      continue;
    Value *Minterm =
        B.CreateAnd(Index & 0b100 ? Src0 : Not0, Index & 0b010 ? Src1 : Not1);
    Minterm =
        B.CreateAnd(Minterm, Index & 0b001 ? Src2 : Not2, "bitop3.minterm");
    Result = Result ? B.CreateOr(Result, Minterm, "bitop3") : Minterm;
  }
  // An empty truth table selects no minterm and so produces zero.
  return Result ? Result : ConstantInt::get(Src0->getType(), 0);
}

Error writeDestination16(RaiseContext &Ctx, OperandResolver &Op,
                         Value *Result) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();

  IntegerType *I32Ty = Ctx.B.getInt32Ty();
  Value *Bits = Ctx.B.CreateZExt(Result, I32Ty, "result.i32");
  if (!Op.dstKeepsOtherHalf()) {
    Ctx.registers().writeReg32(*Dst, Bits);
    return Error::success();
  }

  // LLVM IR has no partial register write, so read the destination back and
  // merge the result into the half the instruction selects.
  Expected<Value *> Previous = Op.dstValue();
  if (!Previous)
    return Previous.takeError();
  Value *Merged;
  if (Op.dstIsHighHalf()) {
    Value *KeptLow = Ctx.B.CreateAnd(
        *Previous,
        ConstantInt::get(
            I32Ty, APInt::getLowBitsSet(RegisterWidthInBits, HalfWidthInBits)));
    Value *Shifted = Ctx.B.CreateShl(Bits, HalfWidthInBits);
    Merged = Ctx.B.CreateOr(KeptLow, Shifted, "merge.hi");
  } else {
    Value *KeptHigh = Ctx.B.CreateAnd(
        *Previous,
        ConstantInt::get(I32Ty, APInt::getHighBitsSet(RegisterWidthInBits,
                                                      HalfWidthInBits)));
    Merged = Ctx.B.CreateOr(KeptHigh, Bits, "merge.lo");
  }
  Ctx.registers().writeReg32(*Dst, Merged);
  return Error::success();
}

Value *applyFloat16SignModifiers(IRBuilder<> &B, Value *Bits, unsigned Mods) {
  Type *I16Ty = B.getInt16Ty();
  if (Mods & SISrcMods::ABS)
    Bits = B.CreateAnd(
        Bits,
        ConstantInt::get(I16Ty, APInt::getSignedMaxValue(HalfWidthInBits)),
        "abs16");
  if (Mods & SISrcMods::NEG)
    Bits = B.CreateXor(
        Bits, ConstantInt::get(I16Ty, APInt::getSignMask(HalfWidthInBits)),
        "neg16");
  return Bits;
}

Error raiseBinary16(RaiseContext &Ctx, OperandResolver &Op,
                    BinaryBuilder Build) {
  Expected<Value *> Src0 = Op.src16(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = Op.src16(1);
  if (!Src1)
    return Src1.takeError();
  return writeDestination16(Ctx, Op, Build(Ctx.B, *Src0, *Src1));
}

Value *maskShiftAmount(IRBuilder<> &B, Value *Amount, unsigned Width) {
  return B.CreateAnd(Amount, ConstantInt::get(Amount->getType(), Width - 1),
                     "shift_amount");
}

Error raiseBitMask(RaiseContext &Ctx, OperandResolver &Op) {
  return raiseBinary32(
      Ctx, Op, [](IRBuilder<> &B, Value *Width, Value *Offset) {
        Width = maskShiftAmount(B, Width, 32);
        Offset = maskShiftAmount(B, Offset, 32);
        Value *HighBit = B.CreateShl(B.getInt32(1), Width);
        Value *Ones = B.CreateSub(HighBit, B.getInt32(1), "bfm.ones");
        return B.CreateShl(Ones, Offset, "bfm");
      });
}

Error raiseBitCount(RaiseContext &Ctx, OperandResolver &Op) {
  return raiseBinary32(Ctx, Op, [](IRBuilder<> &B, Value *Src0, Value *Src1) {
    Value *Count =
        B.CreateUnaryIntrinsic(Intrinsic::ctpop, Src0, nullptr, "bcnt");
    return B.CreateAdd(Count, Src1, "bcnt.add");
  });
}

Error raiseShiftLeft64(RaiseContext &Ctx, OperandResolver &Op) {
  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Amount = Op.src(0);
  if (!Amount)
    return Amount.takeError();
  Expected<Value *> Operand = Op.src64(1);
  if (!Operand)
    return Operand.takeError();
  Value *MaskedAmount = maskShiftAmount(Ctx.B, *Amount, 64);
  Value *Shift = Ctx.B.CreateZExt(MaskedAmount, Ctx.B.getInt64Ty(), "shift64");
  Value *Result = Ctx.B.CreateShl(*Operand, Shift, "lshl64");
  Ctx.registers().writeReg64(*Dst, Result);
  return Error::success();
}

} // namespace COMGR::transpiler
