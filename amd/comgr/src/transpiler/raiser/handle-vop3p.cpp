//===- handle-vop3p.cpp - Transpiler -------------------------------------===//
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
#include "transpiler/raiser/handle-vop-shared.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/wmma-lowering.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {
namespace {

/// Bits of the shift count a 16-bit shift reads.
constexpr uint16_t ShiftCountMask = HalfWidthInBits - 1;

/// Read one two-lane packed floating-point source and apply its lane controls.
Expected<Value *> readPackedFloatSource(RaiseContext &Ctx,
                                        const DecodedInst &Di,
                                        OperandResolver &Op, unsigned Source,
                                        Type *ElementType) {
  assert((ElementType->isHalfTy() || ElementType->isFloatTy()) &&
         "unsupported packed floating-point element type");
  constexpr unsigned AllowedModifiers = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
  unsigned Modifiers = Op.srcMod(Source);
  if (Modifiers & ~AllowedModifiers)
    return unsupported(Ctx, Di, "unsupported packed source modifier");

  FixedVectorType *VectorType = FixedVectorType::get(ElementType, 2);
  Value *NaturalLow;
  Value *NaturalHigh;
  bool IsVectorRegister = false;
  if (ElementType->isFloatTy()) {
    Expected<std::optional<ParsedReg>> SourceReg = Op.srcReg(Source);
    if (!SourceReg)
      return SourceReg.takeError();
    if (*SourceReg)
      IsVectorRegister = (*SourceReg)->RegKind == ParsedReg::VGPR ||
                         (*SourceReg)->RegKind == ParsedReg::AGPR;
  }

  Expected<Value *> SourceBits =
      IsVectorRegister ? Op.src64(Source) : Op.src(Source);
  if (!SourceBits)
    return SourceBits.takeError();

  if (ElementType->isFloatTy() && !IsVectorRegister) {
    // Packed F32 scalar sources provide one value for both result channels.
    Value *Scalar = Ctx.B.CreateBitCast(*SourceBits, ElementType, "pk.scalar");
    NaturalLow = Scalar;
    NaturalHigh = Scalar;
  } else {
    // Packed F16 sources and F32 vector sources carry both channel values.
    IntegerType *ElementIntegerType =
        ElementType->isFloatTy() ? Ctx.B.getInt32Ty() : Ctx.B.getInt16Ty();
    unsigned ElementBitWidth = ElementIntegerType->getBitWidth();
    Value *LowBits =
        Ctx.B.CreateTrunc(*SourceBits, ElementIntegerType, "pk.lo.bits");
    Value *ShiftedBits =
        Ctx.B.CreateLShr(*SourceBits, ElementBitWidth, "pk.hi.shifted");
    Value *HighBits =
        Ctx.B.CreateTrunc(ShiftedBits, ElementIntegerType, "pk.hi.bits");
    NaturalLow = Ctx.B.CreateBitCast(LowBits, ElementType, "pk.lo");
    NaturalHigh = Ctx.B.CreateBitCast(HighBits, ElementType, "pk.hi");
  }

  Value *Low = Modifiers & SISrcMods::OP_SEL_0 ? NaturalHigh : NaturalLow;
  Value *High = Modifiers & SISrcMods::OP_SEL_1 ? NaturalHigh : NaturalLow;
  if (Modifiers & SISrcMods::NEG)
    Low = Ctx.B.CreateFNeg(Low, "pk.neg.lo");
  if (Modifiers & SISrcMods::NEG_HI)
    High = Ctx.B.CreateFNeg(High, "pk.neg.hi");

  Value *Result = PoisonValue::get(VectorType);
  Result = Ctx.B.CreateInsertElement(Result, Low, uint64_t{0}, "pk.insert.lo");
  return Ctx.B.CreateInsertElement(Result, High, 1, "pk.insert.hi");
}

/// Raise packed floating-point arithmetic instructions.
Error raisePackedFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op, Type *ElementType,
                             CanonicalOp Kind) {
  unsigned NumSources = Kind == CanonicalOp::V_PK_FMA_F32 ? 3 : 2;
  assert((Di.NumDefs == 1 && Di.numOperands() != 0 && Di.isReg(0) &&
          Op.nSrcs() == NumSources) &&
         "decoded packed float instruction has unexpected operands");

  if (Error Err = Ctx.validateFPEnvironment(Di, ElementType))
    return Err;

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<Value *> Source0 =
      readPackedFloatSource(Ctx, Di, Op, 0, ElementType);
  if (!Source0)
    return Source0.takeError();
  Expected<Value *> Source1 =
      readPackedFloatSource(Ctx, Di, Op, 1, ElementType);
  if (!Source1)
    return Source1.takeError();

  Value *Result;
  if (Kind == CanonicalOp::V_PK_FMA_F32) {
    Expected<Value *> Source2 =
        readPackedFloatSource(Ctx, Di, Op, 2, ElementType);
    if (!Source2)
      return Source2.takeError();
    Result = Ctx.B.CreateIntrinsic(Intrinsic::fma, {(*Source0)->getType()},
                                   {*Source0, *Source1, *Source2}, nullptr,
                                   "pk.fma");
  } else if (Kind == CanonicalOp::V_PK_ADD_F16 ||
             Kind == CanonicalOp::V_PK_ADD_F32) {
    Result = Ctx.B.CreateFAdd(*Source0, *Source1, "pk.add");
  } else {
    Result = Ctx.B.CreateFMul(*Source0, *Source1, "pk.mul");
  }
  if (*Clamp) {
    FixedVectorType *VectorType = FixedVectorType::get(ElementType, 2);
    Function *Maximum = Intrinsic::getOrInsertDeclaration(
        Ctx.B.GetInsertBlock()->getModule(), Intrinsic::maxnum, {VectorType});
    Function *Minimum = Intrinsic::getOrInsertDeclaration(
        Ctx.B.GetInsertBlock()->getModule(), Intrinsic::minnum, {VectorType});
    Constant *Zero = ConstantVector::getSplat(
        ElementCount::getFixed(2), ConstantFP::get(ElementType, 0.0));
    Constant *One = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantFP::get(ElementType, 1.0));
    Result = Ctx.B.CreateCall(Maximum, {Result, Zero}, "pk.clamp.low");
    Result = Ctx.B.CreateCall(Minimum, {Result, One}, "pk.clamp");
  }

  if (ElementType->isFloatTy()) {
    Ctx.registers().writeRegVec(*Destination, Result);
  } else {
    Value *Packed = Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty(), "pk.pack");
    Ctx.registers().writeReg32(*Destination, Packed);
  }
  return Error::success();
}

/// Read one two-lane packed 16-bit integer source and apply its lane controls.
/// The neg modifiers are defined on the floating-point interpretation only, so
/// a source carrying one is refused.
Expected<Value *> readPackedInt16Source(RaiseContext &Ctx,
                                        const DecodedInst &Di,
                                        OperandResolver &Op, unsigned Source) {
  constexpr unsigned AllowedModifiers =
      SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
  unsigned Modifiers = Op.srcMod(Source);
  if (Modifiers & ~AllowedModifiers)
    return unsupported(Ctx, Di, "unsupported packed integer source modifier");

  Expected<Value *> Bits = Op.src(Source);
  if (!Bits)
    return Bits.takeError();

  IntegerType *ElementType = Ctx.B.getInt16Ty();
  Value *Low = Ctx.B.CreateTrunc(*Bits, ElementType, "pk.lo");
  Value *Shifted = Ctx.B.CreateLShr(*Bits, HalfWidthInBits, "pk.hi.shifted");
  Value *High = Ctx.B.CreateTrunc(Shifted, ElementType, "pk.hi");

  Value *Result = PoisonValue::get(FixedVectorType::get(ElementType, 2));
  Result = Ctx.B.CreateInsertElement(
      Result, Modifiers & SISrcMods::OP_SEL_0 ? High : Low, uint64_t{0},
      "pk.insert.lo");
  return Ctx.B.CreateInsertElement(
      Result, Modifiers & SISrcMods::OP_SEL_1 ? High : Low, 1, "pk.insert.hi");
}

/// Splat `Value` across both lanes of a packed 16-bit vector.
Constant *packedInt16Splat(IRBuilder<> &B, uint16_t Value) {
  return ConstantVector::getSplat(ElementCount::getFixed(2), B.getInt16(Value));
}

/// Raise the packed 16-bit integer operations, which apply the same operation
/// to both halves of each source register.
Error raisePackedInt16(RaiseContext &Ctx, const DecodedInst &Di,
                       OperandResolver &Op) {
  bool IsTernary = Di.CanonOp == CanonicalOp::V_PK_MAD_U16 ||
                   Di.CanonOp == CanonicalOp::V_PK_MAD_I16 ||
                   Di.CanonOp == CanonicalOp::V_PK_MIN3_I16 ||
                   Di.CanonOp == CanonicalOp::V_PK_MAX3_I16 ||
                   Di.CanonOp == CanonicalOp::V_PK_MIN3_U16 ||
                   Di.CanonOp == CanonicalOp::V_PK_MAX3_U16;
  unsigned NumSources = IsTernary ? 3 : 2;
  assert(Op.nSrcs() >= NumSources && "packed operation is missing a source");

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();

  SmallVector<Value *, 3> Sources;
  for (unsigned I = 0; I != NumSources; ++I) {
    Expected<Value *> Source = readPackedInt16Source(Ctx, Di, Op, I);
    if (!Source)
      return Source.takeError();
    Sources.push_back(*Source);
  }

  // The shift opcodes take the count in src0 and the value in src1, and read
  // only the four count bits the hardware honours.
  auto RaiseShift = [&](Instruction::BinaryOps Opcode) {
    Value *Amount = Ctx.B.CreateAnd(
        Sources[0], packedInt16Splat(Ctx.B, ShiftCountMask), "pk.shift.amt");
    return Ctx.B.CreateBinOp(Opcode, Sources[1], Amount, "pk.shift");
  };
  auto RaiseTernaryMinMax = [&](Intrinsic::ID ID) {
    Value *First = Ctx.B.CreateBinaryIntrinsic(ID, Sources[0], Sources[1], {},
                                               "pk.minmax3");
    return Ctx.B.CreateBinaryIntrinsic(ID, First, Sources[2], {}, "pk.minmax3");
  };
  auto RaiseAddSub = [&](bool IsSub, bool IsSigned) -> Value * {
    if (*Clamp) {
      Intrinsic::ID ID =
          IsSub ? (IsSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat)
                : (IsSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat);
      return Ctx.B.CreateBinaryIntrinsic(ID, Sources[0], Sources[1], {},
                                         "pk.addsub");
    }
    return IsSub ? Ctx.B.CreateSub(Sources[0], Sources[1], "pk.sub")
                 : Ctx.B.CreateAdd(Sources[0], Sources[1], "pk.add");
  };
  // The product and sum are formed at 32 bits per lane so that a set clamp bit
  // saturates the value the hardware computes rather than a wrapped one.
  auto RaiseMad = [&](bool IsSigned) {
    FixedVectorType *WideType = FixedVectorType::get(Ctx.B.getInt32Ty(), 2);
    auto Widen = [&](Value *V) {
      return IsSigned ? Ctx.B.CreateSExt(V, WideType)
                      : Ctx.B.CreateZExt(V, WideType);
    };
    Value *Src0 = Widen(Sources[0]);
    Value *Src1 = Widen(Sources[1]);
    Value *Src2 = Widen(Sources[2]);
    Value *Product = Ctx.B.CreateMul(Src0, Src1, "pk.mad.mul");
    Value *Sum = Ctx.B.CreateAdd(Product, Src2, "pk.mad.sum");
    if (*Clamp) {
      // Zero-extended lanes cannot sum to a negative value, so only the signed
      // form needs a lower bound.
      if (IsSigned)
        Sum = Ctx.B.CreateBinaryIntrinsic(
            Intrinsic::smax, Sum,
            ConstantVector::getSplat(ElementCount::getFixed(2),
                                     Ctx.B.getInt32(INT16_MIN)));
      Sum = Ctx.B.CreateBinaryIntrinsic(
          IsSigned ? Intrinsic::smin : Intrinsic::umin, Sum,
          ConstantVector::getSplat(
              ElementCount::getFixed(2),
              Ctx.B.getInt32(IsSigned ? INT16_MAX : UINT16_MAX)));
    }
    return Ctx.B.CreateTrunc(Sum, FixedVectorType::get(Ctx.B.getInt16Ty(), 2),
                             "pk.mad");
  };

  Value *Result;
  switch (Di.CanonOp) {
  case CanonicalOp::V_PK_ADD_U16:
    Result = RaiseAddSub(/*IsSub=*/false, /*IsSigned=*/false);
    break;
  case CanonicalOp::V_PK_ADD_I16:
    Result = RaiseAddSub(/*IsSub=*/false, /*IsSigned=*/true);
    break;
  case CanonicalOp::V_PK_SUB_U16:
    Result = RaiseAddSub(/*IsSub=*/true, /*IsSigned=*/false);
    break;
  case CanonicalOp::V_PK_SUB_I16:
    Result = RaiseAddSub(/*IsSub=*/true, /*IsSigned=*/true);
    break;
  case CanonicalOp::V_PK_MAD_U16:
    Result = RaiseMad(/*IsSigned=*/false);
    break;
  case CanonicalOp::V_PK_MAD_I16:
    Result = RaiseMad(/*IsSigned=*/true);
    break;
  default:
    // Every remaining opcode leaves its result unmodified, so a set clamp bit
    // would have no representation.
    if (*Clamp)
      return unsupported(Ctx, Di,
                         "packed integer operation does not define clamp");
    switch (Di.CanonOp) {
    case CanonicalOp::V_PK_MUL_LO_U16:
      Result = Ctx.B.CreateMul(Sources[0], Sources[1], "pk.mul");
      break;
    case CanonicalOp::V_PK_LSHLREV_B16:
      Result = RaiseShift(Instruction::Shl);
      break;
    case CanonicalOp::V_PK_LSHRREV_B16:
      Result = RaiseShift(Instruction::LShr);
      break;
    case CanonicalOp::V_PK_ASHRREV_I16:
      Result = RaiseShift(Instruction::AShr);
      break;
    case CanonicalOp::V_PK_MIN_I16:
      Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smin, Sources[0],
                                           Sources[1], {}, "pk.minmax");
      break;
    case CanonicalOp::V_PK_MAX_I16:
      Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::smax, Sources[0],
                                           Sources[1], {}, "pk.minmax");
      break;
    case CanonicalOp::V_PK_MIN_U16:
      Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umin, Sources[0],
                                           Sources[1], {}, "pk.minmax");
      break;
    case CanonicalOp::V_PK_MAX_U16:
      Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::umax, Sources[0],
                                           Sources[1], {}, "pk.minmax");
      break;
    case CanonicalOp::V_PK_MIN3_I16:
      Result = RaiseTernaryMinMax(Intrinsic::smin);
      break;
    case CanonicalOp::V_PK_MAX3_I16:
      Result = RaiseTernaryMinMax(Intrinsic::smax);
      break;
    case CanonicalOp::V_PK_MIN3_U16:
      Result = RaiseTernaryMinMax(Intrinsic::umin);
      break;
    case CanonicalOp::V_PK_MAX3_U16:
      Result = RaiseTernaryMinMax(Intrinsic::umax);
      break;
    default:
      llvm_unreachable("not a packed 16-bit integer operation");
    }
    break;
  }

  Ctx.registers().writeReg32(
      *Destination, Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty(), "pk.pack"));
  return Error::success();
}

Expected<Value *> readMixedBF16Source(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op, unsigned Source) {
  unsigned Modifiers = Op.srcMod(Source);
  if (Modifiers & SISrcMods::OP_SEL_1) {
    if (Error Err = Ctx.validateBF16InputDenormMode(Di))
      return std::move(Err);
  }
  Expected<Value *> Bits = Op.src(Source);
  if (!Bits)
    return Bits.takeError();

  std::optional<uint32_t> InlineFloatBits;
  if (!Op.isSrcReg(Source) && Di.sizeInBytes() == 8) {
    uint32_t Immediate = static_cast<uint32_t>(Op.srcImm(Source));
    int32_t SignedImmediate = static_cast<int32_t>(Immediate);
    if (!AMDGPU::isInlinableIntLiteral(SignedImmediate)) {
      if (Immediate <= UINT16_MAX &&
          AMDGPU::isInlinableLiteralBF16(static_cast<int16_t>(Immediate), true))
        InlineFloatBits = Immediate << 16;
      else if (AMDGPU::isInlinableLiteral32(SignedImmediate, true))
        InlineFloatBits = Immediate;
    }
  }
  if (InlineFloatBits)
    *Bits = ConstantInt::get(Ctx.B.getInt32Ty(), *InlineFloatBits);

  Value *Result;
  if (Modifiers & SISrcMods::OP_SEL_1) {
    Value *Selected = *Bits;
    // An inline floating constant occupies either BF16 half, while a literal
    // or register supplies its raw 32-bit word.
    if (InlineFloatBits)
      Selected = ConstantInt::get(Ctx.B.getInt32Ty(), *InlineFloatBits >> 16);
    else if (Modifiers & SISrcMods::OP_SEL_0)
      Selected = Ctx.B.CreateLShr(Selected, 16, "mix.hi");
    Value *HalfBits = Ctx.B.CreateTrunc(Selected, Ctx.B.getInt16Ty());
    Value *BF16 = Ctx.B.CreateBitCast(HalfBits, Ctx.B.getBFloatTy());
    Result = Ctx.B.CreateFPExt(BF16, Ctx.B.getFloatTy(), "mix.bf16");
  } else {
    Result = Ctx.B.CreateBitCast(*Bits, Ctx.B.getFloatTy());
  }

  return Op.applyMods(Source, Result);
}

Error raiseFMAMixF32BF16(RaiseContext &Ctx, const DecodedInst &Di,
                         OperandResolver &Op) {
  assert(Di.NumDefs == 1 && Di.isReg(0) && Op.nSrcs() == 3 &&
         "decoded FMA mix instruction has unexpected operands");
  if (Error Err = Ctx.validateFPEnvironment(Di, Ctx.B.getFloatTy()))
    return Err;

  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();
  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();

  Value *Sources[3];
  for (unsigned I = 0; I != 3; ++I) {
    Expected<Value *> Source = readMixedBF16Source(Ctx, Di, Op, I);
    if (!Source)
      return Source.takeError();
    Sources[I] = *Source;
  }
  Value *Result = Ctx.B.CreateIntrinsic(Intrinsic::fma, {Ctx.B.getFloatTy()},
                                        Sources, nullptr, "mix.fma");
  if (*Clamp) {
    Value *Zero = ConstantFP::get(Ctx.B.getFloatTy(), 0.0);
    Value *One = ConstantFP::get(Ctx.B.getFloatTy(), 1.0);
    Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::maxnum, Result, Zero);
    Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::minnum, Result, One);
  }
  Ctx.registers().writeReg32(*Destination,
                             Ctx.B.CreateBitCast(Result, Ctx.B.getInt32Ty()));
  return Error::success();
}

Expected<Value *> readWMMAAccumulator(RaiseContext &Ctx, const DecodedInst &Di,
                                      OperandResolver &Op,
                                      Type *AccumulatorTy) {
  if (Op.nSrcs() < 3)
    return unsupported(Ctx, Di, "WMMA requires an accumulator source");
  Expected<std::optional<ParsedReg>> Source = Op.srcReg(2);
  if (!Source)
    return Source.takeError();
  if (*Source)
    return Ctx.registers().regFile().readRegVec(Ctx.B, **Source, AccumulatorTy);
  if (!Di.isImm(Op.srcIdx(2)) || Di.getImm(Op.srcIdx(2)) != 0)
    return unsupported(Ctx, Di,
                       "only a zero immediate WMMA accumulator is supported");
  return ConstantAggregateZero::get(AccumulatorTy);
}

Error raiseWMMA(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &Op,
                WMMAInputType InputType) {
  if (Ctx.Projection.sourceWaveSize() != 32 ||
      Ctx.Projection.targetWaveSize() != 64)
    return unsupported(Ctx, Di, "WMMA remapping requires wave32 to wave64");
  unsigned RequiredFeature =
      InputType == WMMAInputType::F16 || InputType == WMMAInputType::F32
          ? AMDGPU::FeatureMAIInsts
      : InputType == WMMAInputType::BF16 ? AMDGPU::FeatureGFX90AInsts
                                         : AMDGPU::FeatureGFX940Insts;
  if (!Ctx.Projection.TargetSTI.hasFeature(RequiredFeature))
    return unsupported(Ctx, Di, "target ISA does not support mapped MFMA");
  if (Op.nSrcs() < 3)
    return unsupported(Ctx, Di, "WMMA requires three source operands");

  if (InputType == WMMAInputType::F32) {
    Function *Kernel = Ctx.B.GetInsertBlock()->getParent();
    MDNode *RequiredSize = Kernel->getMetadata("reqd_work_group_size");
    uint64_t Workitems = 1;
    if (RequiredSize)
      for (const MDOperand &Dimension : RequiredSize->operands())
        Workitems *= mdconst::extract<ConstantInt>(Dimension)->getZExtValue();

    // MFMA requires all target lanes to participate.
    if (!RequiredSize || Workitems % Ctx.Projection.targetWaveSize() != 0)
      if (Error Err = Ctx.requirePerWaveExecution(Di))
        return Err;

    // F32 WMMA preserves denormals, so the mapped MFMA must do the same.
    if (Kernel->getDenormalMode(APFloat::IEEEsingle()) !=
        DenormalMode::getIEEE())
      return unsupported(
          Ctx, Di, "F32 WMMA requires preserved input and output denormals");
  }

  if (InputType == WMMAInputType::IU8) {
    if (Op.srcMod(0) != SISrcMods::NEG || Op.srcMod(1) != SISrcMods::NEG ||
        Op.srcMod(2) != 0)
      return unsupported(Ctx, Di,
                         "WMMA IU8 remapping requires signed matrix inputs");
  } else {
    for (unsigned I = 0; I != 3; ++I) {
      if (Op.srcMod(I) != 0)
        return unsupported(Ctx, Di, "WMMA source modifiers are not supported");
    }
  }
  Expected<bool> Clamp = readClamp(Ctx, Di);
  if (!Clamp)
    return Clamp.takeError();
  if (*Clamp)
    return unsupported(Ctx, Di, "WMMA clamp is not supported");

  Expected<ParsedReg> Destination = Op.dst();
  if (!Destination)
    return Destination.takeError();
  Expected<std::optional<ParsedReg>> SourceA = Op.srcReg(0);
  if (!SourceA)
    return SourceA.takeError();
  Expected<std::optional<ParsedReg>> SourceB = Op.srcReg(1);
  if (!SourceB)
    return SourceB.takeError();
  if (!*SourceA || !*SourceB)
    return unsupported(Ctx, Di, "WMMA matrix inputs must be registers");

  Type *InputTy = FixedVectorType::get(Ctx.B.getInt32Ty(),
                                       InputType == WMMAInputType::F32 ? 2 : 8);
  Type *ElementTy =
      InputType == WMMAInputType::IU8 ? Ctx.B.getInt32Ty() : Ctx.B.getFloatTy();
  Type *AccumulatorTy = FixedVectorType::get(ElementTy, 8);
  AllocaRegFile &Registers = Ctx.registers().regFile();
  Value *A = Registers.readRegVec(Ctx.B, **SourceA, InputTy);
  Value *B = Registers.readRegVec(Ctx.B, **SourceB, InputTy);
  Expected<Value *> C = readWMMAAccumulator(Ctx, Di, Op, AccumulatorTy);
  if (!C)
    return C.takeError();
  Ctx.requireKernelEntryExec(Di);
  Expected<Value *> Result = emitWMMAtoMFMA(Ctx, A, B, *C, InputType);
  if (!Result)
    return Result.takeError();
  Ctx.registers().emitWithNonzeroExec([&] {
    Ctx.registers().regFile().writeRegVec(Ctx.B, *Destination, *Result);
  });
  return Error::success();
}

} // namespace

Error handleVOP3P(RaiseContext &Ctx, const DecodedInst &Di,
                  OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::V_PK_ADD_F16:
  case CanonicalOp::V_PK_MUL_F16:
    return raisePackedFloatBinary(Ctx, Di, Op, Ctx.B.getHalfTy(), Di.CanonOp);
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
  case CanonicalOp::V_PK_FMA_F32:
    return raisePackedFloatBinary(Ctx, Di, Op, Ctx.B.getFloatTy(), Di.CanonOp);
  case CanonicalOp::V_PK_ADD_U16:
  case CanonicalOp::V_PK_ADD_I16:
  case CanonicalOp::V_PK_SUB_U16:
  case CanonicalOp::V_PK_SUB_I16:
  case CanonicalOp::V_PK_MUL_LO_U16:
  case CanonicalOp::V_PK_MAD_U16:
  case CanonicalOp::V_PK_MAD_I16:
  case CanonicalOp::V_PK_LSHLREV_B16:
  case CanonicalOp::V_PK_LSHRREV_B16:
  case CanonicalOp::V_PK_ASHRREV_I16:
  case CanonicalOp::V_PK_MIN_I16:
  case CanonicalOp::V_PK_MAX_I16:
  case CanonicalOp::V_PK_MIN_U16:
  case CanonicalOp::V_PK_MAX_U16:
  case CanonicalOp::V_PK_MIN3_I16:
  case CanonicalOp::V_PK_MAX3_I16:
  case CanonicalOp::V_PK_MIN3_U16:
  case CanonicalOp::V_PK_MAX3_U16:
    return raisePackedInt16(Ctx, Di, Op);
  case CanonicalOp::V_FMA_MIX_F32_BF16:
    return raiseFMAMixF32BF16(Ctx, Di, Op);
  case CanonicalOp::V_WMMA_F32_16x16x4_F32:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::F32);
  case CanonicalOp::V_WMMA_F32_16x16x32_F16:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::F16);
  case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::BF16);
  case CanonicalOp::V_WMMA_I32_16x16x64_IU8:
    return raiseWMMA(Ctx, Di, Op, WMMAInputType::IU8);
  default:
    return unsupported(Ctx, Di);
  }
}

} // namespace COMGR::transpiler
