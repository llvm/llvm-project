//===- raise-context.cpp - Transpiler -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/raise-context.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {

Expected<RaiseContext>
RaiseContext::create(IRBuilder<> &B, const WaveProjection &Projection,
                     const MCState &MC, const SetPcAnalysis &SetPc,
                     const DenseSet<uint64_t> &PairedSplitBarriers,
                     const KernelMeta &Meta, ArrayRef<uint8_t> SourceTextBytes,
                     uint64_t SourceTextBaseAddress,
                     ArrayRef<TextSection::ImageSection> SourceImageSections,
                     uint64_t KernelStartOffset, uint64_t KernelEndOffset,
                     std::optional<bool> SourceSramEcc) {
  // SRAM ECC is always enabled on gfx1250, independently of code-object
  // compatibility flags.
  if (AMDGPU::getSubArchFromGPUName(Projection.SourceSTI.getCPU()) ==
      Triple::AMDGPUSubArch1250) {
    if (SourceSramEcc == false)
      return RaiseFailure::general(
          RaiseFailureReason::BadInput,
          "gfx1250 source does not support disabling SRAM ECC");
    SourceSramEcc = true;
  }
  Expected<RegisterState> Registers =
      RegisterState::create(B, Projection, MC, Meta);
  if (!Registers)
    return Registers.takeError();
  const unsigned SourceFloatRoundMode32 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  const unsigned SourceFloatRoundMode16_64 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  const unsigned SourceFloatDenormMode16_64 = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  const bool SourceBF16InputDenormsFlush =
      SourceFloatDenormMode16_64 == amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC_DST ||
      SourceFloatDenormMode16_64 == amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC;
  const bool SourceFp16Overflow = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL);
  bool Dx10Clamp = true;
  bool IeeeMode = true;
  if (Projection.SourceSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
    Dx10Clamp =
        AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                        amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);
    IeeeMode =
        AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                        amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);
  }
  RaiseContext Context(
      B, Projection, MC, SetPc, PairedSplitBarriers, std::move(*Registers),
      SourceTextBytes, SourceTextBaseAddress, SourceImageSections,
      KernelStartOffset, KernelEndOffset, SourceFloatRoundMode32,
      SourceFloatRoundMode16_64, SourceBF16InputDenormsFlush,
      SourceFp16Overflow, Dx10Clamp, IeeeMode);
  Context.SourceSramEcc = SourceSramEcc;
  return Context;
}

void RaiseContext::requireZeroBits(Value *Value, uint32_t Mask,
                                   const DecodedInst &Di, const Twine &Detail) {
  assert(Value->getType()->isIntegerTy(32) && "expected a register word");
  BitRequirements.push_back({Value, Mask, &Di, Detail.str()});
}

Error RaiseContext::validateRequiredBits() const {
  const DataLayout &Layout = B.GetInsertBlock()->getModule()->getDataLayout();
  for (const RequiredBits &Requirement : BitRequirements) {
    assert(Requirement.Value && "required value was deleted before validation");
    KnownBits Bits = computeKnownBits(Requirement.Value, Layout);
    if ((Bits.Zero.getZExtValue() & Requirement.Mask) != Requirement.Mask) {
      const DecodedInst &Di = *Requirement.Instruction;
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags), Requirement.Detail);
    }
  }
  return Error::success();
}

void RaiseContext::requireWaveUniform(Value *Operand, const DecodedInst &Di,
                                      const Twine &Detail) {
  if (Projection.validationKind() ==
          WaveProjection::ValidationKind::WaveNative ||
      Projection.usesReplicatedDispatch())
    UniformityRequirements.push_back({Operand, &Di, Detail.str()});
}

void RaiseContext::requireKernelEntryExec(const DecodedInst &Di) {
  if (Projection.validationKind() ==
          WaveProjection::ValidationKind::WaveNative ||
      Projection.usesReplicatedDispatch())
    EntryExecRequirements.push_back(
        {Registers.readExec(), &Di,
         "cannot prove that source EXEC at this instruction matches its value "
         "at kernel entry"});
}

void RaiseContext::requireScalarControlFlow(Instruction *Branch,
                                            const DecodedInst &Di) {
  ScalarControlFlowRequirements.push_back(
      {Branch, &Di,
       Projection.allowsDivergentScalarControlFlow()
           ? "source-wave branches must reconverge before a workgroup barrier"
           : "projection requires scalar control flow uniform across the "
             "target wave"});
}

Error RaiseContext::requirePerWaveExecution(const DecodedInst &Di) const {
  if (Projection.validationKind() == WaveProjection::ValidationKind::WaveNative)
    return RaiseFailure::atInstruction(
        RaiseFailureReason::RequiresPerSourceWaveExecution,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "instruction requires a separate target wave for each source wave");
  return Error::success();
}

Error RaiseContext::validateWaveRequirements(TargetMachine &TM,
                                             Value *KernelEntryExec) const {
  for (const RequiredValue &Requirement : EntryExecRequirements) {
    assert(Requirement.Operand &&
           "required value was deleted before validation");
    if (Requirement.Operand == KernelEntryExec)
      continue;
    // For whole-wave replicated launches, all-ones is the entry mask.
    const ConstantInt *Entry = dyn_cast<ConstantInt>(KernelEntryExec);
    if (Projection.usesReplicatedDispatch() && Entry && Entry->isMinusOne() &&
        computeKnownBits(Requirement.Operand,
                         B.GetInsertBlock()->getModule()->getDataLayout())
            .isAllOnes())
      continue;
    const DecodedInst &Di = *Requirement.Instruction;
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnprovenKernelEntryExec,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags), Requirement.Detail);
  }

  if (UniformityRequirements.empty() && ScalarControlFlowRequirements.empty())
    return Error::success();

  Function &F = *B.GetInsertBlock()->getParent();
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return CycleAnalysis(); });
  FAM.registerPass([&] {
    return TargetIRAnalysis(
        [&](const Function &F) { return TM.getTargetTransformInfo(F); });
  });
  FAM.registerPass([&] { return UniformityInfoAnalysis(); });
  // Register promotion exposes scalar data flow across source blocks.
  const UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  PostDominatorTree PDT(F);
  for (const RequiredValue &Requirement : ScalarControlFlowRequirements) {
    Instruction *Branch = cast<Instruction>(Requirement.Operand);
    const DecodedInst &Di = *Requirement.Instruction;
    BasicBlock *Block = Branch->getParent();
    // A non-returning instruction can leave decoded branches unreachable.
    if (!DT.isReachableFromEntry(Block) || !UI.isDivergentTerminator(Branch))
      continue;
    if (!Projection.allowsDivergentScalarControlFlow())
      return RaiseFailure::atInstruction(
          RaiseFailureReason::NonUniformScalarState,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags), Requirement.Detail);
    DomTreeNodeBase<BasicBlock> *Node = PDT.getNode(Block);
    BasicBlock *Join =
        Node && Node->getIDom() ? Node->getIDom()->getBlock() : nullptr;
    SmallVector<BasicBlock *> Worklist(successors(Block));
    SmallPtrSet<BasicBlock *, 32> Visited;
    while (!Worklist.empty()) {
      BasicBlock *Current = Worklist.pop_back_val();
      if (Current == Join || !Visited.insert(Current).second)
        continue;
      for (Instruction &I : *Current) {
        const IntrinsicInst *Call = dyn_cast<IntrinsicInst>(&I);
        if (Call && Call->getIntrinsicID() == Intrinsic::amdgcn_s_barrier)
          return RaiseFailure::atInstruction(
              RaiseFailureReason::NonUniformScalarState,
              strippedMnemonic(MC, Di.Inst), Di.Offset,
              formatName(Di.TargetSpecificFlags), Requirement.Detail);
      }
      append_range(Worklist, successors(Current));
    }
  }
  for (const RequiredValue &Requirement : UniformityRequirements) {
    assert(Requirement.Operand &&
           "required value was deleted before validation");
    Value *Operand = Requirement.Operand;
    if (!UI.isDivergentAtDef(Operand) &&
        (!Operand->hasUseList() || none_of(Operand->uses(), [&](const Use &U) {
          return UI.isDivergentAtUse(U);
        })))
      continue;
    const DecodedInst &Di = *Requirement.Instruction;
    return RaiseFailure::atInstruction(
        RaiseFailureReason::NonUniformScalarState,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags), Requirement.Detail);
  }
  return Error::success();
}

RaiseContext::RaiseContext(
    IRBuilder<> &B, const WaveProjection &Projection, const MCState &MC,
    const SetPcAnalysis &SetPc, const DenseSet<uint64_t> &PairedSplitBarriers,
    RegisterState Registers, ArrayRef<uint8_t> SourceTextBytes,
    uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset,
    unsigned SourceFloatRoundMode32, unsigned SourceFloatRoundMode16_64,
    bool SourceBF16InputDenormsFlush, bool SourceFp16Overflow,
    bool SourceDx10Clamp, bool SourceIeeeMode)
    : B(B), Projection(Projection), MC(MC), SetPc(SetPc),
      PairedSplitBarriers(PairedSplitBarriers), Registers(std::move(Registers)),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset),
      SourceFloatRoundMode32(SourceFloatRoundMode32),
      SourceFloatRoundMode16_64(SourceFloatRoundMode16_64),
      SourceBF16InputDenormsFlush(SourceBF16InputDenormsFlush),
      SourceFp16Overflow(SourceFp16Overflow), SourceDx10Clamp(SourceDx10Clamp),
      SourceIeeeMode(SourceIeeeMode) {}

Error RaiseContext::validateHardwareEffect(const DecodedInst &Di) const {
  if (!Projection.usesReplicatedDispatch() &&
      !Projection.allowsDivergentScalarControlFlow())
    return Error::success();
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedWaveProjection,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      "replicated dispatch does not support per-wave hardware effects");
}

Error RaiseContext::validateFPEnvironment(const DecodedInst &Di,
                                          Type *Ty) const {
  assert((Ty->isHalfTy() || Ty->isFloatTy() || Ty->isDoubleTy()) &&
         "unsupported floating-point type");

  if (Ty->isFloatTy() &&
      !Projection.TargetSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
    if (!SourceDx10Clamp) {
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedFloatingPointMode,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          "source DX10_CLAMP=0 is not representable on a target with fixed "
          "DX10 clamp mode");
    }

    if (!SourceIeeeMode) {
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedFloatingPointMode,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          "source IEEE_MODE=0 is not representable on a target with fixed "
          "IEEE mode");
    }
  }

  if (Ty->isHalfTy() && SourceFp16Overflow) {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedFloatingPointMode,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "FP16 overflow saturation is unsupported");
  }

  unsigned RoundMode =
      Ty->isFloatTy() ? SourceFloatRoundMode32 : SourceFloatRoundMode16_64;
  if (RoundMode != amdhsa::FLOAT_ROUND_MODE_NEAR_EVEN) {
    StringRef TypeName = Ty->isHalfTy()    ? "f16"
                         : Ty->isFloatTy() ? "f32"
                                           : "f64";
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedFloatingPointMode,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine(TypeName) + " rounding mode " + Twine(RoundMode) +
            " is unsupported");
  }

  return Error::success();
}

Error RaiseContext::validateBF16InputDenormMode(const DecodedInst &Di) const {
  if (!SourceBF16InputDenormsFlush)
    return Error::success();
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedFloatingPointMode,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      "BF16 input denormal flushing is unsupported");
}

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  DenseMap<uint64_t, BasicBlock *>::iterator It = OffsetToBb.find(Addr);
  if (It != OffsetToBb.end())
    return It->second;
  // Every branch target is a block leader recorded during CFG layout, so a
  // miss is a raiser bug, not a recoverable case.
  report_fatal_error("transpiler: missing basic block for offset 0x" +
                     Twine::utohexstr(Addr));
}

void RaiseContext::defineBB(uint64_t Addr, BasicBlock *BB) {
  if (!OffsetToBb.try_emplace(Addr, BB).second)
    report_fatal_error("transpiler: duplicate basic block for offset 0x" +
                       Twine::utohexstr(Addr));
}

Value *RaiseContext::emitLaneIdx() { return Projection.emitLaneIdx(B); }

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  if (Projection.sourceWaveSize() != 32 || Projection.targetWaveSize() == 32)
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

} // namespace COMGR::transpiler
