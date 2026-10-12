//===- register-state.cpp - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/register-state.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {

// Where an architected-SGPR prologue leaves the workgroup id.
constexpr unsigned ArchitectedWorkgroupIdXTtmp = 9;
constexpr unsigned ArchitectedWorkgroupIdYZTtmp = 7;
constexpr unsigned ArchitectedWorkgroupIdZBitOffset = 16;

Expected<RegisterState> RegisterState::create(IRBuilder<> &B,
                                              const WaveProjection &Projection,
                                              const MCState &MC,
                                              const KernelMeta &Meta) {
  UserSgprLayout Layout;
  if (Error Err = UserSgprLayout::tryFromKernelMeta(
          Meta, Projection.SourceSTI, MC.SubtargetInfo->getCPU(), Layout))
    return std::move(Err);
  RegisterState Registers(B, Projection, MC, std::move(Layout));
  if (Error Err = Registers.seedEntrySgprs(Meta))
    return std::move(Err);
  Registers.seedEntryVgprs(Meta);
  return Registers;
}

// Seed the VGPRs the source ABI preloads before entry. Only the workitem id
// arrives in a VGPR, and compute_pgm_rsrc2 says how many of its dimensions the
// kernel asked for; the projection turns them into the value the source
// expects to read, which on a widening raise is not the target's own id.
void RegisterState::seedEntryVgprs(const KernelMeta &Meta) {
  const unsigned NumDims =
      AMDHSA_BITS_GET(Meta.ComputePgmRsrc2,
                      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID) +
      1;
  assert(NumDims <= 3 && "workitem id has three dimensions at most");
  if (Projection.SourceSTI.hasFeature(AMDGPU::FeaturePackedTID)) {
    Regs.storeVGPR32(B, 0, Projection.emitPackedWorkitemId(B, NumDims));
    return;
  }
  // One dimension per VGPR, starting at v0, in x, y, z order.
  for (unsigned Dim = 0; Dim < NumDims; ++Dim)
    Regs.storeVGPR32(B, Dim, Projection.emitWorkitemId(B, Dim));
}

// Seed the SGPRs the source ABI preloads before entry with the target
// intrinsics that produce the same values. The layout, not a fixed SGPR
// numbering, says where each source lands: kernarg preload and the
// enable_sgpr_* toggles legally move them.
Error RegisterState::seedEntrySgprs(const KernelMeta &Meta) {
  // A cluster launch hands the cluster id over the TTMPs the workgroup id
  // otherwise arrives in, and the id within the cluster over TTMP6, neither of
  // which the seeding below reproduces.
  if (Meta.hasNonDisabledClusterDims())
    return RaiseFailure::general(
        RaiseFailureReason::UnsupportedSourceClusterDims,
        "the kernel declares cluster dimensions, so its entry state is not the "
        "one the raise can reproduce on the target");

  Module &M = *B.GetInsertBlock()->getModule();
  auto Seed = [&](std::optional<unsigned> Sgpr, Intrinsic::ID Id, bool Is64,
                  const Twine &Name) {
    if (!Sgpr)
      return;
    CallInst *V =
        B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Id), {}, Name);
    if (Projection.usesReplicatedDispatch() &&
        (Id == Intrinsic::amdgcn_dispatch_ptr ||
         Id == Intrinsic::amdgcn_queue_ptr ||
         Id == Intrinsic::amdgcn_dispatch_id))
      UnavailableEntryValues.push_back(V);
    if (Is64)
      Regs.storeSGPR64(B, *Sgpr, V);
    else
      Regs.storeSGPR32(B, *Sgpr, V);
  };

  Seed(Layout.dispatchPtrSgpr(), Intrinsic::amdgcn_dispatch_ptr, true,
       "dispatch_ptr");
  Seed(Layout.queuePtrSgpr(), Intrinsic::amdgcn_queue_ptr, true, "queue_ptr");
  // One kernarg segment pointer serves both the SGPR pair the ABI dedicates to
  // it and the preloaded dwords read back out of the segment below.
  Value *KernargSegment = nullptr;
  if (Layout.needsKernargSegmentPtr())
    KernargSegment =
        B.CreateCall(Intrinsic::getOrInsertDeclaration(
                         &M, Intrinsic::amdgcn_kernarg_segment_ptr),
                     {}, "kernarg_ptr");
  if (std::optional<unsigned> Sgpr = Layout.kernargSegmentPtrSgpr())
    Regs.storeSGPR64(B, *Sgpr, KernargSegment);
  Seed(Layout.dispatchIdSgpr(), Intrinsic::amdgcn_dispatch_id, true,
       "dispatch_id");
  Seed(Layout.workgroupIdXSgpr(), Intrinsic::amdgcn_workgroup_id_x, false,
       "workgroup_id_x");
  Seed(Layout.workgroupIdYSgpr(), Intrinsic::amdgcn_workgroup_id_y, false,
       "workgroup_id_y");
  Seed(Layout.workgroupIdZSgpr(), Intrinsic::amdgcn_workgroup_id_z, false,
       "workgroup_id_z");

  // An architected-SGPR prologue delivers the workgroup id in TTMPs instead of
  // entry SGPRs.
  if (Projection.SourceSTI.hasFeature(AMDGPU::FeatureArchitectedSGPRs)) {
    auto Call = [&](Intrinsic::ID Id, const Twine &Name) {
      return B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Id), {}, Name);
    };
    Value *IdZ =
        B.CreateShl(Call(Intrinsic::amdgcn_workgroup_id_z, "workgroup_id_z"),
                    ArchitectedWorkgroupIdZBitOffset);
    writeReg32(ParsedReg{ParsedReg::TTMP, ArchitectedWorkgroupIdXTtmp},
               Call(Intrinsic::amdgcn_workgroup_id_x, "workgroup_id_x"));
    writeReg32(
        ParsedReg{ParsedReg::TTMP, ArchitectedWorkgroupIdYZTtmp},
        B.CreateOr(Call(Intrinsic::amdgcn_workgroup_id_y, "workgroup_id_y"),
                   IdZ, "workgroup_id_yz"));
  }

  // Recreate source kernarg preloads by loading each dword from its recorded
  // offset in the kernarg segment and seeding the corresponding source SGPR.
  for (auto [Index, LayoutEntry] : enumerate(Layout.Entries)) {
    if (LayoutEntry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
      continue;
    Value *ByteOffset = B.getInt64(LayoutEntry.KernargByteOffset);
    Value *Address = B.CreateInBoundsGEP(B.getInt8Ty(), KernargSegment,
                                         ByteOffset, "preload_gep");
    Value *Dword =
        B.CreateAlignedLoad(B.getInt32Ty(), Address, Align(4), "preload_dw");
    Regs.storeSGPR32(B, static_cast<unsigned>(Index), Dword);
  }

  // No target intrinsic reproduces the remaining entry sources, which carry
  // source private-segment and packed dispatch state. Refuse rather than leave
  // them unseeded: a handler would read an undef SGPR as if it held real entry
  // state.
  for (auto [Index, LayoutEntry] : enumerate(Layout.Entries)) {
    switch (LayoutEntry.SrcKind) {
    case UserSgprLayout::Source::PrivateSegmentBuffer:
    case UserSgprLayout::Source::FlatScratchInit:
    case UserSgprLayout::Source::PrivateSegmentSize:
    case UserSgprLayout::Source::WorkgroupInfo:
      return RaiseFailure::general(
          RaiseFailureReason::UnsupportedEntrySgprSource,
          "s" + Twine(Index) +
              " holds an entry source the raise cannot "
              "reproduce on the target");
    default:
      break;
    }
  }
  return Error::success();
}

Error RegisterState::validateEntrySgprs() {
  if (UnavailableEntryValues.empty())
    return Error::success();
  // Fold SGPR splitting and joining before following pointer uses.
  Function &F = *B.GetInsertBlock()->getParent();
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  InstCombinePass().run(F, FAM);
  for (WeakTrackingVH &Handle : UnavailableEntryValues) {
    CallInst *Seed = dyn_cast_or_null<CallInst>(Handle);
    if (!Seed)
      continue;
    SmallVector<Instruction *> Worklist{Seed};
    SmallVector<std::pair<LoadInst *, unsigned>> DispatchLoads;
    SmallPtrSet<Instruction *, 32> Visited;
    Visited.insert(Seed);
    for (size_t I = 0; I != Worklist.size(); ++I) {
      for (User *U : Worklist[I]->users()) {
        Instruction *Use = cast<Instruction>(U);
        if (!Visited.insert(Use).second)
          continue;
        if (LoadInst *Load = dyn_cast<LoadInst>(Use)) {
          using namespace PatternMatch;
          ConstantInt *Offset = nullptr;
          unsigned Bits = Load->getType()->getPrimitiveSizeInBits();
          if (Seed->getIntrinsicID() == Intrinsic::amdgcn_dispatch_ptr &&
              Load->isSimple() && Load->getType()->isIntOrIntVectorTy() &&
              match(Load->getPointerOperand(),
                    m_IntToPtr(m_Add(m_PtrToInt(m_Specific(Seed)),
                                     m_ConstantInt(Offset)))) &&
              Offset->getValue().ult(DispatchWorkgroupSizesEndOffset) &&
              Offset->getZExtValue() >= DispatchWorkgroupSizeXOffset &&
              Bits % 8 == 0 &&
              Bits <=
                  (DispatchWorkgroupSizesEndOffset - Offset->getZExtValue()) *
                      8) {
            DispatchLoads.emplace_back(Load, Offset->getZExtValue());
            continue;
          }
        }
        if (Use->mayReadOrWriteMemory() ||
            !wouldInstructionBeTriviallyDead(Use))
          return RaiseFailure::general(
              RaiseFailureReason::UnsupportedEntrySgprSource,
              "replicated dispatch cannot reproduce consumed entry state '" +
                  Seed->getCalledFunction()->getName() + "'");
        Worklist.push_back(Use);
      }
    }
    for (const auto &[Load, Offset] : DispatchLoads) {
      IRBuilder<> Builder(Load);
      Value *Sizes = Builder.getInt64(0);
      for (unsigned Dim = 0; Dim != 3; ++Dim) {
        Value *Size = Projection.emitWorkgroupSize(Builder, Dim);
        Size = Builder.CreateZExt(Size, Builder.getInt64Ty());
        Size =
            Builder.CreateShl(Size, Dim * DispatchWorkgroupSizeFieldBytes * 8);
        Sizes = Builder.CreateOr(Sizes, Size);
      }
      // The packet's reserved word following Z is zero.
      Sizes = Builder.CreateLShr(Sizes,
                                 (Offset - DispatchWorkgroupSizeXOffset) * 8);
      Type *BitsType =
          Builder.getIntNTy(Load->getType()->getPrimitiveSizeInBits());
      Value *Bits = Builder.CreateZExtOrTrunc(Sizes, BitsType);
      Load->replaceAllUsesWith(Builder.CreateBitCast(Bits, Load->getType()));
      Load->eraseFromParent();
    }
    for (Instruction *I : Worklist)
      I->dropAllReferences();
    for (Instruction *I : Worklist)
      I->eraseFromParent();
  }
  return Error::success();
}

RegisterState::RegisterState(IRBuilder<> &B, const WaveProjection &Projection,
                             const MCState &MC, UserSgprLayout Layout)
    : B(B), Projection(Projection), MC(MC), Layout(std::move(Layout)) {
  Regs.init(B, B.getInt32Ty(), B.getInt1Ty(), Projection.SourceSTI, *MC.RegInfo,
            Projection);

  SgprShadows.reserve(Regs.Sgpr.size());
  for (unsigned I = 0, E = Regs.Sgpr.size(); I != E; ++I) {
    AllocaInst *WaveMask = B.CreateAlloca(Projection.execStorageTy(), nullptr,
                                          "sgpr_mask_shadow_" + Twine(I));
    AllocaInst *WaveMaskValid =
        B.CreateAlloca(B.getInt1Ty(), nullptr, "sgpr_mask_valid_" + Twine(I));
    AllocaInst *WaveMaskIsPair =
        B.CreateAlloca(B.getInt1Ty(), nullptr, "sgpr_mask_is_pair_" + Twine(I));
    AllocaInst *SourceWavePair = B.CreateAlloca(
        B.getInt64Ty(), nullptr, "source_wave_sgpr_pair_" + Twine(I));
    AllocaInst *SourceWavePairValid = B.CreateAlloca(
        B.getInt1Ty(), nullptr, "source_wave_sgpr_pair_valid_" + Twine(I));
    B.CreateStore(ConstantInt::get(Projection.execStorageTy(), 0), WaveMask);
    B.CreateStore(B.getFalse(), WaveMaskValid);
    B.CreateStore(B.getFalse(), WaveMaskIsPair);
    B.CreateStore(B.getInt64(0), SourceWavePair);
    B.CreateStore(B.getFalse(), SourceWavePairValid);
    SgprShadows.push_back({WaveMask, WaveMaskValid, WaveMaskIsPair,
                           SourceWavePair, SourceWavePairValid});
  }
}

void RegisterState::computeVGPRAdjust(const DecodedInst &Di) {
  unsigned Opc = Di.Inst.getOpcode();
  const MCInstrDesc &Desc = MC.InstrInfo->get(Opc);
  CurrentVgprAdjust.assign(std::max(Di.numOperands(), Desc.getNumOperands()),
                           0u);
  const uint8_t MsBs = blockState().VgprMsBs;
  if (MsBs == 0)
    return;

  // Operand slots are format-specific rather than positional.
  const VGPRMSBOperandIndices OperandIndices = getVGPRMSBOperandIndices(Desc);
  for (unsigned Slot = 0; Slot != OperandIndices.size(); ++Slot) {
    unsigned Adjust =
        ((static_cast<unsigned>(MsBs) >> (Slot * 2)) & 0x3u) * 256u;
    if (Adjust == 0)
      continue;
    auto [XOperandIndex, YOperandIndex] = OperandIndices[Slot];
    auto RecordAdjustment = [&](std::optional<unsigned> OperandIndex) {
      if (!OperandIndex)
        return;
      if (*OperandIndex >= CurrentVgprAdjust.size())
        llvm_unreachable("VGPR operand index exceeds instruction operands");
      CurrentVgprAdjust[*OperandIndex] = Adjust;
    };
    RecordAdjustment(XOperandIndex);
    RecordAdjustment(YOperandIndex);
  }
}

// Return Reg's position in RC, or std::nullopt if it is not a member.
static std::optional<unsigned> findIndexInClass(const MCRegisterClass &RC,
                                                MCRegister Reg) {
  for (unsigned I = 0, E = RC.getNumRegs(); I != E; ++I)
    if (RC.getRegister(I) == Reg)
      return I;
  return std::nullopt;
}

// Return the selected 32-bit register half from the current source-wave mask.
static Value *emitSourceWaveMask32(IRBuilder<> &B,
                                   const WaveProjection &Projection,
                                   Value *Mask, ParsedReg Reg,
                                   const Twine &Name) {
  IntegerType *I32Ty = B.getInt32Ty();
  Mask = Projection.emitCurrentSourceWaveMask(B, Mask, Name);
  if (Mask->getType() == I32Ty) {
    return Mask;
  }
  if (Reg.WidthInDwords < 2 && Reg.BaseIdx == 1) {
    Mask = B.CreateLShr(Mask, 32, Name + "_hi_shr");
  }
  return B.CreateTrunc(
      Mask, I32Ty,
      Reg.WidthInDwords < 2 && Reg.BaseIdx == 1 ? Name + "_hi" : Name + "_lo");
}

// Return whether the current source wave's mask is empty.
static Value *emitSourceWaveMaskIsZero(IRBuilder<> &B,
                                       const WaveProjection &Projection,
                                       Value *Mask, const Twine &Name) {
  Mask = Projection.emitCurrentSourceWaveMask(B, Mask, Name + "_mask");
  return B.CreateICmpEQ(Mask, ConstantInt::get(Mask->getType(), 0), Name);
}

Expected<ParsedReg> RegisterState::parseReg(const DecodedInst &Di,
                                            unsigned OperandIndex) const {
  assert(OperandIndex < Di.numOperands() && "operand index out of range");
  assert(Di.isReg(OperandIndex) && "operand must be a register");
  MCRegister Reg = Di.getReg(OperandIndex);
  ParsedReg Pr;
  if (!Reg) {
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  }

  const MCRegisterInfo &MRI = *MC.RegInfo;
  const MCRegister CanonicalReg = stripRegEncoding(Reg);
  const auto RegisterFailure = [&](const Twine &Detail) -> Error {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine("register-decode: ") + Detail);
  };

  const MCInstrDesc &Descriptor = MC.InstrInfo->get(Di.Inst.getOpcode());
  if (OperandIndex >= Descriptor.getNumOperands())
    llvm_unreachable("register operand has no instruction descriptor");
  const MCOperandInfo &OperandInfo = Descriptor.operands()[OperandIndex];
  if (OperandInfo.RegClass == -1)
    llvm_unreachable("register operand has no register class");
  const int16_t RegisterClassID = MC.InstrInfo->getOpRegClassID(
      OperandInfo,
      MC.SubtargetInfo->getHwMode(MCSubtargetInfo::HwMode_RegInfo));
  if (RegisterClassID < 0)
    llvm_unreachable("register class lookup failed");
  const MCRegisterClass &RegisterClass =
      MRI.getRegClass(static_cast<unsigned>(RegisterClassID));
  if (!RegisterClass.contains(CanonicalReg) && !isInlineValue(CanonicalReg))
    return RegisterFailure(Twine("register '") + MRI.getName(Reg) +
                           "' is not in operand register class '" +
                           MRI.getRegClassName(&RegisterClass) + "'");
  const unsigned WidthInDwords = divideCeil(RegisterClass.getSizeInBits(), 32u);

  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = stripRegEncoding(Lane);

  switch (Lane) {
  case AMDGPU::VCC_HI:
    // VCC_HI is a scratch scalar, not part of VCC, on wave32.
    if (Projection.sourceWaveSize() == 32) {
      Pr.RegKind = ParsedReg::VCC_HI_SCRATCH;
      Pr.WidthInDwords = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::VCC_LO:
    Pr.RegKind = ParsedReg::VCC;
    Pr.BaseIdx = (Lane == AMDGPU::VCC_HI) ? 1 : 0;
    Pr.WidthInDwords =
        CanonicalReg == AMDGPU::VCC
            ? static_cast<uint8_t>(Projection.sourceWaveSize() / 32)
            : 1;
    return Pr;
  case AMDGPU::EXEC_HI:
    // EXEC_HI is a scratch scalar, not part of EXEC, on wave32.
    if (Projection.sourceWaveSize() == 32) {
      Pr.RegKind = ParsedReg::EXEC_HI_SCRATCH;
      Pr.WidthInDwords = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::EXEC_LO:
    Pr.RegKind = ParsedReg::EXEC;
    Pr.BaseIdx = (Lane == AMDGPU::EXEC_HI) ? 1 : 0;
    Pr.WidthInDwords = WidthInDwords;
    return Pr;
  case AMDGPU::SCC:
    Pr.RegKind = ParsedReg::SCC;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::MODE:
    Pr.RegKind = ParsedReg::MODE;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::M0:
    Pr.RegKind = ParsedReg::M0;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    Pr.RegKind = ParsedReg::FLAT_SCR;
    Pr.BaseIdx = (Lane == AMDGPU::FLAT_SCR_HI) ? 1 : 0;
    Pr.WidthInDwords = CanonicalReg == AMDGPU::FLAT_SCR ? 2 : 1;
    return Pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot -- treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    return RegisterFailure(Twine("unsupported register '") + MRI.getName(Reg) +
                           "'");
  // LDS_DIRECT (src_lds_direct, enc 254): reads a dword from LDS at the
  // byte offset held in M0. Used as a VALU source after buffer_load_*_lds.
  case AMDGPU::LDS_DIRECT:
    Pr.RegKind = ParsedReg::LDS_DIRECT;
    Pr.WidthInDwords = 1;
    return Pr;
  // Source-only predicates have no backing register-file slot.
  case AMDGPU::SRC_VCCZ:
    Pr.RegKind = ParsedReg::SRC_VCCZ;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::SRC_EXECZ:
    Pr.RegKind = ParsedReg::SRC_EXECZ;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::SRC_SCC:
    Pr.RegKind = ParsedReg::SRC_SCC;
    Pr.WidthInDwords = 1;
    return Pr;
  // Runtime-defined aperture registers have no static IR representation.
  case AMDGPU::SRC_SHARED_BASE_LO:
  case AMDGPU::SRC_SHARED_LIMIT_LO:
  case AMDGPU::SRC_PRIVATE_BASE_LO:
  case AMDGPU::SRC_PRIVATE_LIMIT_LO:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_LO:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_HI:
    return RegisterFailure(Twine("unsupported register '") + MRI.getName(Reg) +
                           "'");
  default:
    break;
  }

  // The hardware encoding identifies vector and accumulator register families.
  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned HwIdx = Enc & AMDGPU::HWEncoding::REG_IDX_MASK;

  if (Enc & AMDGPU::HWEncoding::IS_AGPR) {
    Pr.RegKind = ParsedReg::AGPR;
    Pr.WidthInDwords = WidthInDwords;
    if (OperandIndex < CurrentVgprAdjust.size())
      HwIdx += CurrentVgprAdjust[OperandIndex];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }
  if (Enc & AMDGPU::HWEncoding::IS_VGPR) {
    Pr.RegKind = ParsedReg::VGPR;
    Pr.WidthInDwords = WidthInDwords;
    if (OperandIndex < CurrentVgprAdjust.size())
      HwIdx += CurrentVgprAdjust[OperandIndex];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }

  // TTMP encodings vary by generation; class position is the stable index.
  const MCRegisterClass &TTMP32 = MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (std::optional<unsigned> Index = findIndexInClass(TTMP32, Lane)) {
    Pr.RegKind = ParsedReg::TTMP;
    Pr.BaseIdx = *Index;
    Pr.WidthInDwords = WidthInDwords;
    return Pr;
  }

  // The broader SReg_32 class also contains architectural registers handled
  // above, so classify general SGPRs through SGPR_32.
  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane)) {
    Pr.RegKind = ParsedReg::SGPR;
    Pr.BaseIdx = HwIdx;
    Pr.WidthInDwords = WidthInDwords;
    return Pr;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("register-decode: could not classify register '") +
          MRI.getName(Reg) + "' (enc=0x" + Twine::utohexstr(Enc) + ")");
}

void RegisterState::storeSCC(IRBuilder<> &B, Value *V) {
  Regs.storeSCC(B, V);
  blockState().DefinedScc = true;
  blockState().SccFromSourceImage = false;
}

void RegisterState::storeSourceImageSCC(IRBuilder<> &B, Value *V) {
  Regs.storeSCC(B, V);
  blockState().DefinedScc = true;
  blockState().SccFromSourceImage = true;
  SourceImageSccWritten = true;
}

bool RegisterState::sccMayHoldSourceImageBit() {
  if (blockState().SccFromSourceImage)
    return true;
  return SourceImageSccWritten && !blockState().DefinedScc;
}

Expected<Value *> RegisterState::readSCC(const DecodedInst &Di) {
  if (sccMayHoldSourceImageBit())
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        "operand-read: 'scc' may hold a carry or overflow derived from a "
        "source code-object address; relocation may change the result");
  return Regs.loadSCC(B);
}

bool RegisterState::sourceImageSgprPairRecorded(unsigned Idx) const {
  // A pair is keyed by its low SGPR, so this register belongs to the pair
  // keyed at its own index and to the one before it.
  return SourceImageSgprPairs.contains(Idx) ||
         (Idx > 0 && SourceImageSgprPairs.contains(Idx - 1));
}

Error RegisterState::refuseSourceImageRead(const DecodedInst &Di,
                                           unsigned Idx) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: 's") + Twine(Idx) +
          "' may hold a source code-object address, which names a place in "
          "the source image rather than anything the raised kernel can "
          "address");
}

Error RegisterState::refuseSourceImageRead(const DecodedInst &Di,
                                           unsigned BaseIdx,
                                           unsigned WidthInDwords) {
  for (unsigned I = 0; I != WidthInDwords; ++I) {
    unsigned Idx = BaseIdx + I;
    if (mayHoldSourceImageAddress(Idx))
      return refuseSourceImageRead(Di, Idx);
    // The register carries no address the raise knows of yet, and the block
    // reading it holds whatever ran before rather than a value of its own, so
    // an instruction the raise has still to reach can record one there.
    if (!sourceImageSgprPairRecorded(Idx) &&
        !blockState().DefinedSgprs.contains(Idx))
      DeferredSourceImageReads.push_back({/*Di=*/&Di, /*Idx=*/Idx});
  }
  return Error::success();
}

Error RegisterState::refuseSourceImageRead(const DecodedInst &Di,
                                           const ParsedReg &Pr) {
  if (Pr.RegKind != ParsedReg::SGPR || !Pr.BaseIdx)
    return Error::success();
  return refuseSourceImageRead(Di, *Pr.BaseIdx, Pr.WidthInDwords);
}

Error RegisterState::refuseDeferredSourceImageReads() {
  for (const DeferredSourceImageRead &Read : DeferredSourceImageReads)
    if (sourceImageSgprPairRecorded(Read.Idx))
      return refuseSourceImageRead(*Read.Di, Read.Idx);
  return Error::success();
}

Expected<Value *> RegisterState::readSgpr32(const DecodedInst &Di,
                                            unsigned Idx) {
  if (Error Err = refuseSourceImageRead(Di, Idx, /*WidthInDwords=*/1))
    return Err;
  return Regs.loadSGPR32(B, Idx);
}

Expected<Value *> RegisterState::readSgpr64(const DecodedInst &Di,
                                            unsigned Idx) {
  if (Error Err = refuseSourceImageRead(Di, Idx, /*WidthInDwords=*/2))
    return Err;
  return Regs.loadSGPR64(B, Idx);
}

Expected<Value *> RegisterState::readOp32(const DecodedInst &Di,
                                          unsigned OpIdx) {
  IntegerType *I32Ty = B.getInt32Ty();
  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Error Err = refuseSourceImageRead(Di, Pr))
      return Err;
    if (Pr.RegKind == ParsedReg::VCC) {
      Value *Mask = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
      return emitSourceWaveMask32(B, Projection, Mask, Pr, "vcc_src_wave");
    }
    if (Pr.RegKind == ParsedReg::EXEC) {
      return emitSourceWaveMask32(B, Projection, Regs.loadExec(B), Pr,
                                  "exec_src_wave");
    }
    if (Pr.RegKind == ParsedReg::SCC || Pr.RegKind == ParsedReg::SRC_SCC) {
      Expected<Value *> Scc = readSCC(Di);
      if (!Scc)
        return Scc.takeError();
      return B.CreateZExt(*Scc, I32Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_VCCZ)
      return B.CreateZExt(emitVccIsZero(), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_EXECZ)
      return B.CreateZExt(emitExecIsZero(), I32Ty);
    if (Pr.RegKind == ParsedReg::NOREG)
      return ConstantInt::get(I32Ty, 0);
    if (Pr.RegKind == ParsedReg::MODE)
      return ConstantInt::get(I32Ty, 0);
    Value *V = Regs.readReg32(B, Pr);
    if (!V)
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          Twine("operand-read: could not read 32-bit register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst));
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::get(I32Ty, static_cast<uint32_t>(*Val));
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve 32-bit operand ") + Twine(OpIdx) +
          " in " + strippedMnemonic(MC, Di.Inst));
}

Expected<Value *> RegisterState::readOpSourceWaveMask32(const DecodedInst &Di,
                                                        unsigned OpIdx) {
  if (!Di.isReg(OpIdx))
    return readOp32(Di, OpIdx);

  Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
  if (!Reg)
    return Reg.takeError();
  ParsedReg Pr = *Reg;
  if (Pr.RegKind == ParsedReg::EXEC)
    return Projection.emitCurrentSourceWaveMask(B, Regs.loadExec(B),
                                                "exec_srcwave_mask");
  if (Pr.RegKind == ParsedReg::VCC)
    return Projection.emitCurrentSourceWaveMask(
        B, Regs.readVCCAsWaveMask(B, Projection.execStorageTy()),
        "vcc_srcwave_mask");
  if (Pr.RegKind == ParsedReg::SGPR && Pr.BaseIdx) {
    Expected<Value *> Fallback = readOp32(Di, OpIdx);
    if (!Fallback)
      return Fallback.takeError();
    if (Value *ShadowValid = loadSgprWaveMaskValid(*Pr.BaseIdx)) {
      Value *ShadowExec = loadSgprWaveMaskExec(*Pr.BaseIdx);
      if (ShadowExec->getType() != Projection.execStorageTy())
        ShadowExec = B.CreateZExtOrTrunc(ShadowExec, Projection.execStorageTy(),
                                         "sgpr_mask_exec_cast");
      Value *ShadowMask = Projection.emitCurrentSourceWaveMask(
          B, ShadowExec, "sgpr_srcwave_mask_shadow");
      return B.CreateSelect(ShadowValid, ShadowMask, *Fallback,
                            "sgpr_srcwave_mask");
    }
    return *Fallback;
  }

  return readOp32(Di, OpIdx);
}

Expected<Value *> RegisterState::readOp64(const DecodedInst &Di,
                                          unsigned OpIdx) {
  IntegerType *I64Ty = B.getInt64Ty();
  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Error Err = refuseSourceImageRead(Di, Pr))
      return Err;
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, I64Ty);
    if (Pr.RegKind == ParsedReg::EXEC) {
      Value *V = Regs.loadExec(B);
      if (V->getType() != I64Ty)
        V = B.CreateZExt(V, I64Ty, "exec_ext");
      return V;
    }
    // These unbacked architectural registers read as zero for compute kernels.
    if (Pr.RegKind == ParsedReg::NOREG || Pr.RegKind == ParsedReg::MODE)
      return ConstantInt::get(I64Ty, 0);
    if (Pr.RegKind == ParsedReg::SRC_SCC) {
      Expected<Value *> Scc = readSCC(Di);
      if (!Scc)
        return Scc.takeError();
      return B.CreateZExt(*Scc, I64Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_VCCZ)
      return B.CreateZExt(emitVccIsZero(), I64Ty);
    if (Pr.RegKind == ParsedReg::SRC_EXECZ)
      return B.CreateZExt(emitExecIsZero(), I64Ty);
    Value *V = Regs.readReg64(B, Pr);
    if (!V)
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          Twine("operand-read: could not read 64-bit register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst));
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::getSigned(I64Ty, *Val);
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve 64-bit operand ") + Twine(OpIdx) +
          " in " + strippedMnemonic(MC, Di.Inst));
}

Value *RegisterState::emitLaneActiveBit() {
  // A cached bit still dominates its uses: the lane-active branches emitted
  // while raising one instruction follow one another, each joining back before
  // the next begins. Instruction boundaries and EXEC writes reset the cache.
  if (Value *Cached = blockState().CachedLaneActive)
    return Cached;

  Value *Active = Projection.emitLaneActiveBit(B, Regs.loadExec(B));
  blockState().CachedLaneActive = Active;
  return Active;
}

void RegisterState::writeReg32(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg32(B, Pr, V); });
  } else {
    Regs.writeReg32(B, Pr, V);
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
    else if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      invalidateSgprWaveMaskI1(*Pr.BaseIdx);
    } else if (Pr.RegKind == ParsedReg::M0) {
      updateM0Const(V);
    } else if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx == 8) {
      TTMP8EntryValueAvailable = false;
    }
  }
}

void RegisterState::writeReg64(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg64(B, Pr, V); });
  } else {
    Regs.writeReg64(B, Pr, V);
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
    else if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      invalidateSgprWaveMaskI1(*Pr.BaseIdx);
      invalidateSgprWaveMaskI1(*Pr.BaseIdx + 1);
    } else if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx &&
               *Pr.BaseIdx <= 8 && *Pr.BaseIdx + 1 >= 8) {
      TTMP8EntryValueAvailable = false;
    }
  }
}

void RegisterState::writeRegVec(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeRegVec(B, Pr, V); });
  } else {
    // Vector-valued scalar writes cannot target EXEC.
    Regs.writeRegVec(B, Pr, V);
    if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      for (unsigned I = 0; I != Pr.WidthInDwords; ++I)
        invalidateSgprWaveMaskI1(*Pr.BaseIdx + I);
    } else if (Pr.RegKind == ParsedReg::TTMP && Pr.BaseIdx &&
               *Pr.BaseIdx <= 8 && *Pr.BaseIdx + Pr.WidthInDwords > 8) {
      TTMP8EntryValueAvailable = false;
    }
  }
}

void RegisterState::writeRegExecWidth(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  // Wave-mask writes are wave-level effects and must not be EXEC-predicated.
  Regs.writeRegExecWidth(B, Pr, V);
  if (Pr.RegKind == ParsedReg::EXEC)
    resetLaneActiveCache();
  else if (Pr.RegKind == ParsedReg::SGPR) {
    assert(Pr.BaseIdx && "SGPR must have a base register index");
    unsigned WidthInDwords =
        Projection.sourceWaveScopedLaneOps() && Pr.WidthInDwords >= 2
            ? 2
            : Projection.sourceWaveMaskTy()
                      ->getPrimitiveSizeInBits()
                      .getFixedValue() /
                  32;
    for (unsigned I = 0; I != WidthInDwords; ++I)
      invalidateSgprWaveMaskI1(*Pr.BaseIdx + I);
  }
}

void RegisterState::writeReg32UnderExec(ParsedReg Pr,
                                        function_ref<Value *()> Body) {
  // Only a per-lane register can take a predicated write. A scalar one commits
  // for the whole wave, and writing it invalidates facts that a body running
  // under EXEC cannot be trusted to have invalidated.
  assert((Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) &&
         "predicated register write must target a per-lane register");
  emitUnderExec([&] { Regs.writeReg32(B, Pr, Body()); });
}

void RegisterState::storeVGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR32(B, Idx, V); });
}

void RegisterState::storeVGPR64(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR64(B, Idx, V); });
}

void RegisterState::storeAGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeAGPR32(B, Idx, V); });
}

void RegisterState::emitUnderExec(llvm::function_ref<void()> Body) {
  emitUnderCondition(emitLaneActiveBit(), Body);
}

void RegisterState::emitMemoryEffect(function_ref<void()> Body) {
  if (!Projection.usesReplicatedDispatch()) {
    Body();
    return;
  }
  Value *Lane = Projection.emitLaneIdx(B);
  Value *Primary = B.CreateICmpULT(
      Lane, B.getInt32(Projection.sourceWaveSize()), "primary_lane");
  emitUnderCondition(Primary, Body);
}

Value *RegisterState::emitMemoryValue(function_ref<Value *()> Body,
                                      bool IsScalar) {
  bool PackedScalar = IsScalar && Projection.allowsDivergentScalarControlFlow();
  if (!Projection.usesReplicatedDispatch() && !PackedScalar)
    return Body();

  Value *Lane = Projection.emitLaneIdx(B);
  Value *SourceLane =
      PackedScalar
          ? B.CreateAnd(Lane, B.getInt32(Projection.sourceWaveSize() - 1))
          : nullptr;
  Value *Primary =
      PackedScalar
          ? B.CreateICmpEQ(SourceLane, B.getInt32(0))
          : B.CreateICmpULT(
                Lane, B.getInt32(IsScalar ? 1 : Projection.sourceWaveSize()),
                "primary_lane");
  BasicBlock *Before = B.GetInsertBlock();
  Value *Result = nullptr;
  BasicBlock *ResultBlock = nullptr;
  emitUnderCondition(Primary, [&] {
    Result = Body();
    ResultBlock = B.GetInsertBlock();
    assert(Result && !ResultBlock->hasTerminator() &&
           "memory read must produce a value and fall through");
  });
  Type *ResultType = Result->getType();
  PHINode *Merged = B.CreatePHI(ResultType, 2, "memory_result");
  Merged->addIncoming(Constant::getNullValue(ResultType), Before);
  Merged->addIncoming(Result, ResultBlock);

  unsigned BitWidth = ResultType->getPrimitiveSizeInBits();
  assert((ResultType->isIntegerTy() || isa<FixedVectorType>(ResultType)) &&
         (BitWidth <= 32 || BitWidth % 32 == 0) &&
         "memory result must fit in whole register words");
  unsigned NumWords = divideCeil(BitWidth, 32u);
  Type *WordsType = NumWords == 1
                        ? static_cast<Type *>(B.getInt32Ty())
                        : FixedVectorType::get(B.getInt32Ty(), NumWords);
  Value *Words = BitWidth < 32 ? B.CreateZExt(Merged, WordsType)
                               : B.CreateBitCast(Merged, WordsType);
  if (PackedScalar)
    SourceLane = B.CreateSub(Lane, SourceLane, "scalar_memory_source_lane");
  else
    SourceLane = B.CreateAnd(Lane, B.getInt32(Projection.sourceWaveSize() - 1));
  Value *Selector = B.CreateShl(SourceLane, 2);
  Value *Broadcast = PoisonValue::get(WordsType);
  for (unsigned I = 0; I != NumWords; ++I) {
    Value *Word = NumWords == 1 ? Words : B.CreateExtractElement(Words, I);
    // Each selected owner participates in this gather.
    Word = IsScalar && !PackedScalar
               ? B.CreateIntrinsic(Intrinsic::amdgcn_readlane, {B.getInt32Ty()},
                                   {Word, B.getInt32(0)})
               : B.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {},
                                   {Selector, Word});
    Broadcast =
        NumWords == 1 ? Word : B.CreateInsertElement(Broadcast, Word, I);
  }
  return BitWidth < 32 ? B.CreateTrunc(Broadcast, ResultType)
                       : B.CreateBitCast(Broadcast, ResultType);
}

void RegisterState::emitWithNonzeroExec(llvm::function_ref<void()> Body) {
  Value *ExecNonzero = B.CreateNot(emitExecIsZero(), "exec_nonzero");
  emitUnderCondition(ExecNonzero, Body);
}

void RegisterState::emitUnderCondition(Value *Condition,
                                       llvm::function_ref<void()> Body) {
  BasicBlock *PreBb = B.GetInsertBlock();
  Function *F = PreBb->getParent();
  BasicBlock *DoBb = BasicBlock::Create(B.getContext(), "spe_do", F);
  BasicBlock *SkipBb = BasicBlock::Create(B.getContext(), "spe_skip", F);
  B.CreateCondBr(Condition, DoBb, SkipBb);

  // This splits one source block across several LLVM blocks, all of them
  // dominated by the block the state was established in, so the state survives
  // them. The guarded path is the exception: what it defines does not dominate
  // the join, so the join continues from the state that path started with.
  const BlockState Entry = blockState();
  B.SetInsertPoint(DoBb);
  carryStateIntoCurrentBlock();
  Body();
  // Body may terminate its block; do not add a second terminator.
  if (!B.GetInsertBlock()->hasTerminator())
    B.CreateBr(SkipBb);

  B.SetInsertPoint(SkipBb);
  State = Entry;
  carryStateIntoCurrentBlock();
}

Value *RegisterState::emitExecIsZero() {
  return emitSourceWaveMaskIsZero(B, Projection, Regs.loadExec(B), "execz");
}

Value *RegisterState::emitVccIsZero() {
  Value *Vcc = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
  return emitSourceWaveMaskIsZero(B, Projection, Vcc, "vccz");
}

Expected<Value *> RegisterState::readOpExecWidth(const DecodedInst &Di,
                                                 unsigned OpIdx) {
  auto WidenToExec = [&](Value *Narrow) -> Value * {
    Type *ExecTy = Projection.execStorageTy();
    if (Narrow->getType() == ExecTy)
      return Narrow;
    unsigned Have = Narrow->getType()->getPrimitiveSizeInBits();
    unsigned Want = ExecTy->getPrimitiveSizeInBits();
    if (Have >= Want)
      return B.CreateZExtOrTrunc(Narrow, ExecTy);
    Value *Zext = B.CreateZExt(Narrow, ExecTy, "wn_src_to_exec_zext");
    Value *Hi = B.CreateShl(Zext, Have);
    return B.CreateOr(Zext, Hi, "wn_src_to_exec_mask");
  };

  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Error Err = refuseSourceImageRead(Di, Pr))
      return Err;
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
    if (Pr.RegKind == ParsedReg::EXEC)
      return Regs.loadExec(B);
    if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ||
        Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
      // Wave32 vcc_hi / exec_hi are scratch scalars, not the wave mask.
      return WidenToExec(Regs.readReg32(B, Pr));
    if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      unsigned BaseIdx = *Pr.BaseIdx;
      Value *Narrow =
          (Projection.sourceWaveScopedLaneOps() && Pr.WidthInDwords >= 2)
              ? Regs.loadSGPR64(B, BaseIdx)
              : (Projection.sourceWaveSize() == 32
                     ? Regs.loadSGPR32(B, BaseIdx)
                     : Regs.loadSGPR64(B, BaseIdx));
      Value *Fallback = WidenToExec(Narrow);
      if (Value *ShadowValid = loadSgprWaveMaskValid(BaseIdx)) {
        Value *ShadowExec = loadSgprWaveMaskExec(BaseIdx);
        if (ShadowExec->getType() != Projection.execStorageTy())
          ShadowExec = B.CreateZExtOrTrunc(
              ShadowExec, Projection.execStorageTy(), "wm_shadow_exec_cast");
        return B.CreateSelect(ShadowValid, ShadowExec, Fallback,
                              "exec_width_sgpr_shadow_sel");
      }
      return Fallback;
    }
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine("operand-read: could not read EXEC-width register '") +
            MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
            strippedMnemonic(MC, Di.Inst));
  }
  // Interpret immediate masks at source width and replicate them like SGPR
  // operands when widening.
  const unsigned SourceWaveSize = Projection.sourceWaveSize();
  Type *SrcTy = SourceWaveSize == 32 ? B.getInt32Ty() : B.getInt64Ty();
  uint64_t SrcMask =
      SourceWaveSize == 32 ? 0xFFFFFFFFull : 0xFFFFFFFFFFFFFFFFull;
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    uint64_t Bits = static_cast<uint64_t>(*Val) & SrcMask;
    Value *Narrow = ConstantInt::get(SrcTy, Bits, /*IsSigned=*/false);
    return WidenToExec(Narrow);
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve EXEC-width operand ") +
          Twine(OpIdx) + " in " + strippedMnemonic(MC, Di.Inst));
}

Expected<Value *> RegisterState::readOpWaveMaskI1(const DecodedInst &Di,
                                                  unsigned OpIdx) {
  if (!Di.isReg(OpIdx)) {
    Expected<Value *> Mask = readOpExecWidth(Di, OpIdx);
    if (!Mask)
      return Mask.takeError();
    return Projection.extractLaneBitFromWaveMask(B, *Mask);
  }

  Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
  if (!Reg)
    return Reg.takeError();
  ParsedReg Pr = *Reg;
  switch (Pr.RegKind) {
  case ParsedReg::SGPR: {
    assert(Pr.BaseIdx && "SGPR must have a base register index");
    return lookupSgprWaveMaskI1(*Pr.BaseIdx, Pr.WidthInDwords >= 2);
  }
  case ParsedReg::VCC:
    return Regs.loadVCC(B);
  case ParsedReg::EXEC:
    return Projection.extractLaneBitFromWaveMask(B, Regs.loadExec(B));
  default:
    return nullptr;
  }
}

void RegisterState::recordSgprWaveMaskI1(unsigned BaseIdx, Value *CmpI1,
                                         bool IsPair) {
  blockState().LastSgprWaveMaskI1[BaseIdx] = WaveMaskEntry{CmpI1, IsPair};
  if (BaseIdx < SgprShadows.size()) {
    Value *ExecMask = Projection.ballotI1ToWidth(
        B, CmpI1, Projection.execStorageTy(), "wm_shadow_exec");
    B.CreateStore(ExecMask, SgprShadows[BaseIdx].WaveMask);
    B.CreateStore(B.getTrue(), SgprShadows[BaseIdx].WaveMaskValid);
    B.CreateStore(B.getInt1(IsPair), SgprShadows[BaseIdx].WaveMaskIsPair);
  }
}

void RegisterState::recordWaveMaskI1(ParsedReg Dst, Value *MaskI1) {
  if (!MaskI1)
    return;
  switch (Dst.RegKind) {
  case ParsedReg::SGPR:
    if (Dst.BaseIdx)
      recordSgprWaveMaskI1(*Dst.BaseIdx, MaskI1,
                           /*IsPair=*/Dst.WidthInDwords >= 2);
    return;
  case ParsedReg::VCC:
    Regs.storeVCC(B, MaskI1);
    return;
  case ParsedReg::EXEC: {
    Value *Mask = Projection.ballotI1ToWidth(
        B, MaskI1, Projection.execStorageTy(), "wave_mask_exec");
    storeExec(Mask);
    return;
  }
  default:
    return;
  }
}

Value *RegisterState::emitCurrentSourceWaveHasActiveLane() {
  Value *Exec = Regs.loadExec(B);
  if (!Projection.providesSourceWaveExecInvariant())
    return emitLaneActiveBit();
  return B.CreateNot(emitSourceWaveMaskIsZero(B, Projection, Exec, "execz"),
                     "source_wave_active");
}

void RegisterState::recordSourceWaveSgprPair(unsigned BaseIdx, Value *V) {
  if (!Projection.providesSourceWaveExecInvariant()) {
    return;
  }
  if (BaseIdx >= SgprShadows.size()) {
    return;
  }
  const SgprShadow &Shadow = SgprShadows[BaseIdx];
  Value *Old = B.CreateLoad(B.getInt64Ty(), Shadow.SourceWavePair,
                            "source_wave_sgpr_pair_old");
  Value *OldValid = B.CreateLoad(B.getInt1Ty(), Shadow.SourceWavePairValid,
                                 "source_wave_sgpr_pair_valid_old");
  Value *Active = emitCurrentSourceWaveHasActiveLane();
  Value *Merged = B.CreateSelect(Active, V, Old, "source_wave_sgpr_pair");
  Value *Valid = B.CreateSelect(Active, B.getTrue(), OldValid,
                                "source_wave_sgpr_pair_valid");
  B.CreateStore(Merged, Shadow.SourceWavePair);
  B.CreateStore(Valid, Shadow.SourceWavePairValid);
}

Value *RegisterState::materializeSourceWaveSgprPair(unsigned BaseIdx,
                                                    Value *Fallback) {
  if (!Projection.providesSourceWaveExecInvariant() ||
      BaseIdx >= SgprShadows.size()) {
    return Fallback;
  }
  const SgprShadow &Shadow = SgprShadows[BaseIdx];
  Value *Recorded = B.CreateLoad(B.getInt64Ty(), Shadow.SourceWavePair,
                                 "source_wave_sgpr_pair");
  Value *Valid = B.CreateLoad(B.getInt1Ty(), Shadow.SourceWavePairValid,
                              "source_wave_sgpr_pair_valid");
  return B.CreateSelect(Valid, Recorded, Fallback, "source_wave_sgpr_pair_sel");
}

Value *RegisterState::loadSgprWaveMaskExec(unsigned BaseIdx) const {
  if (BaseIdx >= SgprShadows.size()) {
    return nullptr;
  }
  return B.CreateLoad(Projection.execStorageTy(), SgprShadows[BaseIdx].WaveMask,
                      "sgpr_mask_exec");
}

Value *RegisterState::loadSgprWaveMaskValid(unsigned BaseIdx) const {
  if (BaseIdx >= SgprShadows.size()) {
    return nullptr;
  }
  return B.CreateLoad(B.getInt1Ty(), SgprShadows[BaseIdx].WaveMaskValid,
                      "sgpr_mask_valid");
}

void RegisterState::invalidateSgprWaveMaskI1(unsigned BaseIdx) {
  blockState().LastSgprWaveMaskI1.erase(BaseIdx);
  blockState().SourceImageSgprPairAddrShadow.erase(BaseIdx);
  blockState().DefinedSgprs.insert(BaseIdx);
  if (BaseIdx < SgprShadows.size()) {
    B.CreateStore(B.getFalse(), SgprShadows[BaseIdx].WaveMaskValid);
    B.CreateStore(B.getFalse(), SgprShadows[BaseIdx].SourceWavePairValid);
  }
  if (BaseIdx > 0) {
    DenseMap<unsigned, WaveMaskEntry> &Recorded =
        blockState().LastSgprWaveMaskI1;
    DenseMap<unsigned, WaveMaskEntry>::iterator Prev =
        Recorded.find(BaseIdx - 1);
    if (Prev != Recorded.end() && Prev->second.IsPair) {
      Recorded.erase(Prev);
    }
    if (BaseIdx - 1 < SgprShadows.size()) {
      const SgprShadow &Previous = SgprShadows[BaseIdx - 1];
      Value *PreviousValid = B.CreateLoad(B.getInt1Ty(), Previous.WaveMaskValid,
                                          "sgpr_mask_previous_valid");
      Value *PreviousIsPair = B.CreateLoad(
          B.getInt1Ty(), Previous.WaveMaskIsPair, "sgpr_mask_previous_is_pair");
      Value *KeepPrevious =
          B.CreateAnd(PreviousValid, B.CreateNot(PreviousIsPair),
                      "sgpr_mask_keep_previous");
      B.CreateStore(KeepPrevious, Previous.WaveMaskValid);
      B.CreateStore(B.getFalse(), SgprShadows[BaseIdx - 1].SourceWavePairValid);
    }
    blockState().SourceImageSgprPairAddrShadow.erase(BaseIdx - 1);
  }
}

std::optional<uint64_t>
RegisterState::lookupSourceImageSgprPairAddr(unsigned BaseIdx) {
  DenseMap<unsigned, uint64_t> &Recorded =
      blockState().SourceImageSgprPairAddrShadow;
  DenseMap<unsigned, uint64_t>::const_iterator It = Recorded.find(BaseIdx);
  if (It == Recorded.end())
    return std::nullopt;
  return It->second;
}

std::optional<bool> RegisterState::takeSourceImageCarry(unsigned BaseIdx,
                                                        uint64_t Offset) {
  std::optional<BlockState::SourceImageCarryState> &Pending =
      blockState().SourceImageCarry;
  if (!Pending || Pending->PairBaseIdx != BaseIdx ||
      Pending->NextOffset != Offset)
    return std::nullopt;
  bool Carry = Pending->Carry;
  Pending.reset();
  return Carry;
}

bool RegisterState::droppedSourceImageSgprPairAddr(unsigned BaseIdx) {
  if (!SourceImageSgprPairs.contains(BaseIdx) ||
      blockState().SourceImageSgprPairAddrShadow.contains(BaseIdx))
    return false;
  // A block that wrote both halves itself holds what it wrote there, whatever
  // the block that computed the address left in them.
  const DenseSet<unsigned> &Defined = blockState().DefinedSgprs;
  return !Defined.contains(BaseIdx) || !Defined.contains(BaseIdx + 1);
}

bool RegisterState::mayHoldSourceImageAddress(unsigned Idx) {
  // A pair is keyed by its low SGPR, so this register is the low half of the
  // pair keyed at its own index and the high half of the one before it.
  auto PairCarriesAddress = [&](unsigned BaseIdx) {
    if (!SourceImageSgprPairs.contains(BaseIdx))
      return false;
    if (blockState().SourceImageSgprPairAddrShadow.contains(BaseIdx))
      return true;
    // Another block recorded the address, and this one holds what that block
    // left there unless it has written the register itself.
    return !blockState().DefinedSgprs.contains(Idx);
  };
  return PairCarriesAddress(Idx) || (Idx > 0 && PairCarriesAddress(Idx - 1));
}

void RegisterState::updateM0Const(Value *V) {
  ConstantInt *CI = dyn_cast<ConstantInt>(V);
  blockState().M0Const =
      CI ? std::optional<uint64_t>(CI->getZExtValue()) : std::nullopt;
}

RegisterState::BlockState &RegisterState::blockState() {
  BasicBlock *Current = B.GetInsertBlock();
  if (StateBlock != Current) {
    State = BlockState();
    StateBlock = Current;
  }
  return State;
}

void RegisterState::invalidateSgprShadows() {
  for (const SgprShadow &Shadow : SgprShadows) {
    B.CreateStore(B.getFalse(), Shadow.WaveMaskValid);
    B.CreateStore(B.getFalse(), Shadow.SourceWavePairValid);
  }
}

void RegisterState::collectAllocas(SmallVectorImpl<AllocaInst *> &Out) const {
  Regs.collectAllocas(Out);
  for (const SgprShadow &Shadow : SgprShadows) {
    Out.push_back(Shadow.WaveMask);
    Out.push_back(Shadow.WaveMaskValid);
    Out.push_back(Shadow.WaveMaskIsPair);
    Out.push_back(Shadow.SourceWavePair);
    Out.push_back(Shadow.SourceWavePairValid);
  }
}

} // namespace COMGR::transpiler
