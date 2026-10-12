//===- handle-sopk.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>

using namespace llvm;

namespace COMGR::transpiler {
namespace {

// Signed SOPK operations sign extend simm16; unsigned compares zero extend it.
Expected<Value *> readSopkImmediate(RaiseContext &Ctx, const DecodedInst &Di,
                                    bool IsSigned) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                    AMDGPU::OpName::simm16);
  if (Index < 0 || static_cast<unsigned>(Index) >= Di.numOperands() ||
      !Di.isImm(Index))
    return unsupportedInstruction(Ctx, Di, "missing SOPK immediate operand");
  int64_t Raw = Di.getImm(Index);
  if (Raw < INT16_MIN || Raw > UINT16_MAX)
    return unsupportedInstruction(Ctx, Di,
                                  "SOPK immediate exceeds its 16-bit field");
  uint16_t Bits = static_cast<uint16_t>(Raw);
  int32_t Immediate = IsSigned ? SignExtend32<16>(Bits) : Bits;
  return Ctx.B.getInt32(Immediate);
}

bool isUnsignedSopkCompare(CanonicalOp Kind) {
  switch (Kind) {
  case CanonicalOp::S_CMPK_EQ_U32:
  case CanonicalOp::S_CMPK_LG_U32:
  case CanonicalOp::S_CMPK_GT_U32:
  case CanonicalOp::S_CMPK_GE_U32:
  case CanonicalOp::S_CMPK_LT_U32:
  case CanonicalOp::S_CMPK_LE_U32:
    return true;
  default:
    return false;
  }
}

Error handleImmediateSopk(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op) {
  CanonicalOp Kind = Di.CanonOp;
  bool IsUnsignedCompare = isUnsignedSopkCompare(Kind);
  Expected<Value *> Imm = readSopkImmediate(Ctx, Di, !IsUnsignedCompare);
  if (!Imm)
    return Imm.takeError();

  if (Kind == CanonicalOp::S_MOVK_I32) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Ctx.registers().writeReg32(*Dst, *Imm);
    return Error::success();
  }

  if (Op.nSrcs() == 0)
    return unsupportedInstruction(Ctx, Di, "missing SOPK register source");
  Expected<Value *> Source = Op.src(0);
  if (!Source)
    return Source.takeError();

  if (Kind == CanonicalOp::S_ADDK_I32 || Kind == CanonicalOp::S_MULK_I32) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Value *Result;
    if (Kind == CanonicalOp::S_ADDK_I32) {
      Value *Pair = Ctx.B.CreateIntrinsic(
          Intrinsic::sadd_with_overflow, {Ctx.B.getInt32Ty()}, {*Source, *Imm});
      Result = Ctx.B.CreateExtractValue(Pair, 0, "addk");
      Value *Overflow = Ctx.B.CreateExtractValue(Pair, 1, "addk_scc");
      Ctx.registers().storeSCC(Ctx.B, Overflow);
    } else {
      Result = Ctx.B.CreateMul(*Source, *Imm, "mulk");
    }
    Ctx.registers().writeReg32(*Dst, Result);
    return Error::success();
  }

  CmpInst::Predicate Predicate;
  switch (Kind) {
  case CanonicalOp::S_CMPK_EQ_I32:
  case CanonicalOp::S_CMPK_EQ_U32:
    Predicate = CmpInst::ICMP_EQ;
    break;
  case CanonicalOp::S_CMPK_LG_I32:
  case CanonicalOp::S_CMPK_LG_U32:
    Predicate = CmpInst::ICMP_NE;
    break;
  case CanonicalOp::S_CMPK_GT_I32:
    Predicate = CmpInst::ICMP_SGT;
    break;
  case CanonicalOp::S_CMPK_GE_I32:
    Predicate = CmpInst::ICMP_SGE;
    break;
  case CanonicalOp::S_CMPK_LT_I32:
    Predicate = CmpInst::ICMP_SLT;
    break;
  case CanonicalOp::S_CMPK_LE_I32:
    Predicate = CmpInst::ICMP_SLE;
    break;
  case CanonicalOp::S_CMPK_GT_U32:
    Predicate = CmpInst::ICMP_UGT;
    break;
  case CanonicalOp::S_CMPK_GE_U32:
    Predicate = CmpInst::ICMP_UGE;
    break;
  case CanonicalOp::S_CMPK_LT_U32:
    Predicate = CmpInst::ICMP_ULT;
    break;
  case CanonicalOp::S_CMPK_LE_U32:
    Predicate = CmpInst::ICMP_ULE;
    break;
  default:
    return unsupportedInstruction(Ctx, Di);
  }
  Value *Condition = Ctx.B.CreateICmp(Predicate, *Source, *Imm, "cmpk");
  Ctx.registers().storeSCC(Ctx.B, Condition);
  return Error::success();
}

enum class HardwareRegisterReadAction {
  ReturnZero, // Replace the read with zero.
  Reject      // Reject the instruction.
};

enum class HardwareRegisterWriteAction {
  Ignore, // Drop the write.
  Emit,   // Preserve the write with an intrinsic.
  Reject  // Reject the instruction.
};

// Read and write policies for one hardware-register identifier.
struct HardwareRegisterPolicy {
  HardwareRegisterReadAction Read;
  HardwareRegisterWriteAction Write;
};

// Return the conservative policy for hardware-register identifier Id.
// Hardware-register numbers are reused between ISA generations. Where a number
// denotes a load-bearing register on any supported source ISA, refuse the
// access instead of treating it as diagnostic state.
HardwareRegisterPolicy classifyHardwareRegister(unsigned Id) {
  using namespace AMDGPU::Hwreg;
  switch (Id) {
  case ID_MODE:
    // MODE is initialized from the kernel descriptor and can be changed by
    // preceding SETREG instructions. Until that state is tracked, returning a
    // fabricated value would silently change source-visible behavior.
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Emit};

  case ID_MEM_BASES:
  case ID_FLAT_SCR_LO:
  case ID_FLAT_SCR_HI:
  case ID_XNACK_MASK:
  case ID_XNACK_STATE_PRIV:
  case ID_XNACK_MASK_gfx1250:
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Reject};

  case ID_TBA_LO:
  case ID_TBA_HI:
  case ID_TMA_LO:
  case ID_TMA_HI:
    return {HardwareRegisterReadAction::ReturnZero,
            HardwareRegisterWriteAction::Reject};

  case ID_STATUS:
  case ID_TRAPSTS:
  case ID_HW_ID:
  case ID_GPR_ALLOC:
  case ID_LDS_ALLOC:
  case ID_IB_STS:
  case ID_PERF_SNAPSHOT_DATA_gfx12:
  case ID_PERF_SNAPSHOT_PC_LO_gfx12:
  case ID_PERF_SNAPSHOT_PC_HI_gfx12:
  case ID_HW_ID1:
  case ID_HW_ID2:
  case ID_POPS_PACKER:
  case ID_SCHED_MODE:
  case ID_PERF_SNAPSHOT_DATA_gfx11:
  case ID_IB_STS2:
  case ID_SHADER_CYCLES:
  case ID_SHADER_CYCLES_HI:
  case ID_DVGPR_ALLOC_LO:
  case ID_DVGPR_ALLOC_HI:
    return {HardwareRegisterReadAction::ReturnZero,
            HardwareRegisterWriteAction::Ignore};

  default:
    return {HardwareRegisterReadAction::Reject,
            HardwareRegisterWriteAction::Reject};
  }
}

// Read the hardware-register selector from Di.
Expected<unsigned> readHardwareRegisterSelector(RaiseContext &Ctx,
                                                const DecodedInst &Di) {
  unsigned SelectorOperand = 1;
  if (Di.numOperands() <= SelectorOperand || !Di.isImm(SelectorOperand))
    return unsupportedInstruction(
        Ctx, Di, "hardware-register selector is not immediate operand 1");
  return static_cast<unsigned>(Di.getImm(SelectorOperand)) & UINT16_MAX;
}

// Return whether the bit range overlaps MODE.VGPR_MSB.
bool overlapsVgprMsb(unsigned BitOffset, unsigned BitWidth) {
  uint32_t FieldMask = llvm::maskTrailingOnes<uint32_t>(BitWidth) << BitOffset;
  return FieldMask & AMDGPU::Hwreg::VGPR_MSB_MASK;
}

constexpr unsigned VgprMsbFieldWidth =
    llvm::popcount(static_cast<unsigned>(AMDGPU::Hwreg::DST_VGPR_MSB));
constexpr unsigned VgprMsbLowBit = llvm::countr_zero_constexpr<unsigned>(
    static_cast<unsigned>(AMDGPU::Hwreg::VGPR_MSB_MASK));
constexpr unsigned VgprMsbHighBit =
    std::numeric_limits<unsigned>::digits -
    llvm::countl_zero_constexpr<unsigned>(
        static_cast<unsigned>(AMDGPU::Hwreg::VGPR_MSB_MASK)) -
    1;

uint8_t getVgprMsbField(uint64_t Value, uint32_t Mask, unsigned Slot) {
  unsigned Shift = llvm::countr_zero(Mask);
  return static_cast<uint8_t>(((Value & Mask) >> Shift)
                              << (Slot * VgprMsbFieldWidth));
}

// Update the tracked VGPR_MSB state from an immediate MODE value.
void updateImmediateModeVgprMsb(RaiseContext &Ctx, uint64_t Value) {
  uint8_t Encoding = getVgprMsbField(Value, AMDGPU::Hwreg::SRC0_VGPR_MSB, 0) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::SRC1_VGPR_MSB, 1) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::SRC2_VGPR_MSB, 2) |
                     getVgprMsbField(Value, AMDGPU::Hwreg::DST_VGPR_MSB, 3);
  Ctx.registers().setVgprMsBs(Encoding);
}

// Raise a hardware-register read according to Policy.
Error handleGetreg(RaiseContext &Ctx, const DecodedInst &Di,
                   OperandResolver &Op, unsigned Id,
                   HardwareRegisterPolicy Policy) {
  if (Policy.Read == HardwareRegisterReadAction::Reject)
    return unsupportedInstruction(
        Ctx, Di,
        Twine("cannot reproduce hardware-register read for id ") + Twine(Id));

  Expected<ParsedReg> Dst = Op.dst();
  if (!Dst)
    return Dst.takeError();
  Ctx.registers().writeReg32(*Dst, Ctx.B.getInt32(0));
  return Error::success();
}

// Raise a hardware-register write according to Policy.
Error handleSetreg(RaiseContext &Ctx, const DecodedInst &Di,
                   OperandResolver &Op, unsigned Selector, unsigned Id,
                   unsigned BitOffset, unsigned BitWidth,
                   HardwareRegisterPolicy Policy) {
  if (Policy.Write == HardwareRegisterWriteAction::Reject)
    return unsupportedInstruction(
        Ctx, Di,
        Twine("cannot reproduce hardware-register write for id ") + Twine(Id));
  if (Policy.Write == HardwareRegisterWriteAction::Ignore)
    return Error::success();

  unsigned ReplayBit =
      llvm::countr_zero(static_cast<unsigned>(AMDGPU::Hwreg::REPLAY_MODE));
  if (AMDGPU::getSubArchFromGPUName(Ctx.Projection.SourceSTI.getCPU()) ==
          Triple::AMDGPUSubArch1250 &&
      AMDGPU::getSubArchFromGPUName(Ctx.Projection.TargetSTI.getCPU()) ==
          Triple::AMDGPUSubArch942 &&
      Id == AMDGPU::Hwreg::ID_MODE && BitOffset <= ReplayBit &&
      BitWidth > ReplayBit - BitOffset) {
    if (Di.CanonOp == CanonicalOp::S_SETREG_IMM32_B32 &&
        BitOffset == ReplayBit && BitWidth == 1 &&
        Di.Offset == Ctx.kernelStartOffset()) {
      int ImmediateIndex = COMGR::transpiler::getNamedOperandIdx(
          Di.Inst.getOpcode(), AMDGPU::OpName::imm);
      assert(ImmediateIndex >= 0 && Di.isImm(ImmediateIndex) &&
             "expected SETREG immediate");
      // Replay mode constrains source register reuse, not memory visibility.
      // The initial enable has no pending memory; target register allocation
      // and scheduling must satisfy the target's own replay rules.
      if (Di.getImm(ImmediateIndex) == 1)
        return Error::success();
    }
    return unsupportedInstruction(
        Ctx, Di,
        "only immediate REPLAY_MODE enable at kernel entry is supported");
  }

  bool HasExtendedVgprs =
      Ctx.Projection.SourceSTI.hasFeature(AMDGPU::Feature1024AddressableVGPRs);
  Value *ValueArg;
  if (Di.CanonOp == CanonicalOp::S_SETREG_IMM32_B32) {
    uint64_t Value = static_cast<uint64_t>(Di.getImm(0));
    ValueArg = Ctx.B.getInt32(Value);
    if (HasExtendedVgprs && Id == AMDGPU::Hwreg::ID_MODE)
      updateImmediateModeVgprMsb(Ctx, Value);
  } else {
    if (HasExtendedVgprs && Id == AMDGPU::Hwreg::ID_MODE &&
        overlapsVgprMsb(BitOffset, BitWidth))
      return unsupportedInstruction(
          Ctx, Di,
          Twine("dynamic MODE write overlaps VGPR_MSB bits [") +
              Twine(VgprMsbLowBit) + ":" + Twine(VgprMsbHighBit) + "]");
    Expected<Value *> Source = Op.src(0);
    if (!Source)
      return Source.takeError();
    ValueArg = *Source;
  }

  Module *M = Ctx.B.GetInsertBlock()->getModule();
  Function *Setreg =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_setreg);
  Ctx.requireWaveUniform(
      ValueArg, Di, "WaveNative requires uniform hardware register writes");
  if (Error Err = Ctx.validateHardwareEffect(Di))
    return Err;
  Ctx.B.CreateCall(Setreg, {Ctx.B.getInt32(Selector), ValueArg});
  return Error::success();
}

} // namespace

// Raise a supported SOPK instruction.
Error handleSOPK(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::S_MOVK_I32:
  case CanonicalOp::S_ADDK_I32:
  case CanonicalOp::S_MULK_I32:
  case CanonicalOp::S_CMPK_EQ_I32:
  case CanonicalOp::S_CMPK_LG_I32:
  case CanonicalOp::S_CMPK_GT_I32:
  case CanonicalOp::S_CMPK_GE_I32:
  case CanonicalOp::S_CMPK_LT_I32:
  case CanonicalOp::S_CMPK_LE_I32:
  case CanonicalOp::S_CMPK_EQ_U32:
  case CanonicalOp::S_CMPK_LG_U32:
  case CanonicalOp::S_CMPK_GT_U32:
  case CanonicalOp::S_CMPK_GE_U32:
  case CanonicalOp::S_CMPK_LT_U32:
  case CanonicalOp::S_CMPK_LE_U32:
    return handleImmediateSopk(Ctx, Di, Op);
  case CanonicalOp::S_GETREG_B32:
  case CanonicalOp::S_SETREG_B32:
  case CanonicalOp::S_SETREG_IMM32_B32:
    break;
  default:
    return unsupportedInstruction(Ctx, Di);
  }

  Expected<unsigned> Selector = readHardwareRegisterSelector(Ctx, Di);
  if (!Selector)
    return Selector.takeError();
  auto [Id, BitOffset, BitWidth] =
      AMDGPU::Hwreg::HwregEncoding::decode(*Selector);
  HardwareRegisterPolicy Policy = classifyHardwareRegister(Id);

  if (Di.CanonOp == CanonicalOp::S_GETREG_B32)
    return handleGetreg(Ctx, Di, Op, Id, Policy);
  return handleSetreg(Ctx, Di, Op, *Selector, Id, BitOffset, BitWidth, Policy);
}

} // namespace COMGR::transpiler
