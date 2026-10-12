//===- wave-projection.cpp - Transpiler -----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/wave-projection.h"

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/launch.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::transpiler {

unsigned getWaveSize(const MCSubtargetInfo &Subtarget) {
  return Subtarget.hasFeature(AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
}

// ----------------------------------------------------------------------------
// WaveProjection base: lane-id derivation is shared across every projection
// that keeps each target lane mapped 1:1 to a hardware lane.
// ----------------------------------------------------------------------------

WaveProjection::WaveProjection(const MCSubtargetInfo &Source,
                               const MCSubtargetInfo &Target, Type *I32Ty,
                               Type *I64Ty)
    : SourceSTI(Source), TargetSTI(Target), I32Ty(I32Ty), I64Ty(I64Ty),
      ExecStorageTy(getWaveSize(SourceSTI) == 32 ? I32Ty : I64Ty) {
  assert(targetWaveSize() >= sourceWaveSize() &&
         "wave projection does not support narrowing");
}

unsigned WaveProjection::sourceWaveSize() const {
  return getWaveSize(SourceSTI);
}

unsigned WaveProjection::targetWaveSize() const {
  return getWaveSize(TargetSTI);
}

Type *WaveProjection::waveMaskTy() const {
  return targetWaveSize() == 32 ? I32Ty : I64Ty;
}

Type *WaveProjection::sourceWaveMaskTy() const {
  return sourceWaveSize() == 32 ? I32Ty : I64Ty;
}

Value *WaveProjection::emitLaneIdx(IRBuilder<> &B) const {
  // Lane id (mbcnt vs all-ones, base 0) is EXEC-independent and
  // function-invariant: emit it once at a point that dominates the whole
  // function and reuse it everywhere. If nothing consumes it, DCE drops it.
  //
  // The cache is keyed by the function it was emitted into, the kernel boundary
  // this class can observe: a projection reused across kernels re-emits per
  // function instead of returning a value that belongs to a different one.
  Function *F = B.GetInsertBlock()->getParent();
  if (CachedLaneIdx && CachedLaneIdxFunc == F)
    return CachedLaneIdx;

  // Emit in the entry block, after any leading allocas. The allocas must stay
  // at the top of the entry block or mem2reg/SROA may decline to promote them,
  // so insert at the first non-alloca instruction rather than the block start.
  BasicBlock &Entry = F->getEntryBlock();
  IRBuilder<> EB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());

  Module *M = Entry.getModule();
  Type *I32Ty = EB.getInt32Ty();
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_lo);
  Value *AllOnes = ConstantInt::getSigned(I32Ty, -1);
  Value *Zero32 = ConstantInt::get(I32Ty, 0);
  Value *LaneId = EB.CreateCall(MbcntLo, {AllOnes, Zero32}, "lane_lo");
  if (waveMaskTy() != I32Ty) {
    Function *MbcntHi =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_hi);
    LaneId = EB.CreateCall(MbcntHi, {AllOnes, LaneId}, "lane_id");
  }
  CachedLaneIdxFunc = F;
  CachedLaneIdx = LaneId;
  return LaneId;
}

Value *WaveProjection::emitTargetWorkitemId(IRBuilder<> &B,
                                            unsigned Dimension) const {
  assert(Dimension < CachedWorkitemIds.size() && "invalid work-item dimension");
  Function *F = B.GetInsertBlock()->getParent();
  if (CachedWorkitemIds[Dimension])
    return CachedWorkitemIds[Dimension];

  static constexpr Intrinsic::ID Ids[] = {Intrinsic::amdgcn_workitem_id_x,
                                          Intrinsic::amdgcn_workitem_id_y,
                                          Intrinsic::amdgcn_workitem_id_z};
  static constexpr const char *Names[] = {"tid_x", "tid_y", "tid_z"};
  BasicBlock &Entry = F->getEntryBlock();
  IRBuilder<> EB(&Entry, Entry.getFirstNonPHIOrDbgOrAlloca());
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(Entry.getModule(), Ids[Dimension]);
  return CachedWorkitemIds[Dimension] = EB.CreateCall(Fn, {}, Names[Dimension]);
}

Value *WaveProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  return emitTargetWorkitemId(B, 0);
}

Value *WaveProjection::emitWorkitemId(IRBuilder<> &B, unsigned Dim) const {
  return Dim == 0 ? emitWorkitemIdX(B) : emitTargetWorkitemId(B, Dim);
}

Value *WaveProjection::emitTargetWaveId(IRBuilder<> &B) const {
  Module *M = B.GetInsertBlock()->getModule();
  if (TargetSTI.hasFeature(AMDGPU::FeatureArchitectedSGPRs)) {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_wave_id);
    return B.CreateCall(Fn, {}, "target_wave_id");
  }

  Function *DispatchPtrFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_dispatch_ptr);
  Value *DispatchPtr = B.CreateCall(DispatchPtrFn, {}, "dispatch_ptr");
  auto LoadWorkgroupSize = [&](uint64_t Offset, unsigned Dimension,
                               const Twine &Name) {
    Value *Ptr = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), DispatchPtr,
                                              Offset, Name + "_ptr");
    LoadInst *Size = B.CreateAlignedLoad(B.getInt16Ty(), Ptr, Align(2), Name);
    Value *NominalSize = B.CreateZExt(Size, B.getInt32Ty(), Name + "_i32");
    Value *GridPtr =
        B.CreateConstGEP1_64(B.getInt8Ty(), DispatchPtr, 12 + 4 * Dimension);
    Value *GridSize = B.CreateAlignedLoad(B.getInt32Ty(), GridPtr, Align(4));
    Intrinsic::ID GroupId = Dimension == 0 ? Intrinsic::amdgcn_workgroup_id_x
                                           : Intrinsic::amdgcn_workgroup_id_y;
    Value *Group = B.CreateIntrinsic(GroupId, {}, {}, nullptr, "workgroup_id");
    Value *Start = B.CreateMul(Group, NominalSize);
    Value *Remaining = B.CreateSub(GridSize, Start);
    // Edge workgroups can have shorter rows than the dispatch packet's size.
    return B.CreateBinaryIntrinsic(Intrinsic::umin, Remaining, NominalSize);
  };
  Value *SizeX =
      LoadWorkgroupSize(DispatchWorkgroupSizeXOffset, 0, "workgroup_size_x");
  Value *SizeY = LoadWorkgroupSize(DispatchWorkgroupSizeXOffset +
                                       DispatchWorkgroupSizeFieldBytes,
                                   1, "workgroup_size_y");

  Value *X = emitTargetWorkitemId(B, 0);
  Value *Y = emitTargetWorkitemId(B, 1);
  Value *Z = emitTargetWorkitemId(B, 2);
  Value *Row = B.CreateAdd(Y, B.CreateMul(Z, SizeY), "wave_flat_yz");
  Value *FlatId = B.CreateAdd(X, B.CreateMul(Row, SizeX), "wave_flat_id");
  Value *WaveId =
      B.CreateUDiv(FlatId, B.getInt32(targetWaveSize()), "target_wave_id");
  // Workitem IDs in lanes absent at entry need not describe this workgroup.
  return B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {B.getInt32Ty()},
                           {WaveId}, nullptr, "uniform_wave_id");
}

Value *ReplicationProjection::emitSourceWaveId(IRBuilder<> &B) const {
  return emitTargetWaveId(B);
}

Value *ReplicationProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  Value *Raw = WaveProjection::emitWorkitemIdX(B);
  // When the target wave is wider than the source workgroup, the upper target
  // lanes [MaxFlatWG, WaveSize) carry no source workitem, so their raw workitem
  // id is just the hardware lane index. Clamp those lanes to workitem 0 so they
  // replicate lane 0's in-bounds addressing; real lanes are unchanged and every
  // committed result is identical.
  const unsigned SourceWaveSize = sourceWaveSize();
  const unsigned TargetWaveSize = targetWaveSize();
  if (TargetWaveSize > SourceWaveSize && MaxFlatWG > 0 &&
      MaxFlatWG < TargetWaveSize) {
    // The "real lane" test is the flat local id, not workitem.id.x. Under
    // replication the target lane index is the flat local id (lanes are
    // laid out in flat order), so a lane is real iff its index is below the
    // flattened workgroup size. Comparing workitem.id.x against MaxFlatWG would
    // only be correct for 1D workgroups, where tid.x == flat local id; for a
    // multidimensional workgroup tid.x is just the X coordinate while MaxFlatWG
    // is the flattened total.
    Value *Limit = ConstantInt::get(I32Ty, MaxFlatWG);
    Value *FlatLaneId = emitLaneIdx(B);
    Value *IsRealLane = B.CreateICmpULT(FlatLaneId, Limit, "tid_is_real_lane");
    Raw = B.CreateSelect(IsRealLane, Raw, ConstantInt::get(I32Ty, 0),
                         "tid_phantom_clamp");
  }
  return Raw;
}

// Bit offsets of the Y/Z fields in the packed kernel-entry v0 workitem id
// (x[0:9] | y[10:19] | z[20:29])
static constexpr unsigned WorkitemIdYBitOffset = 10;
static constexpr unsigned WorkitemIdZBitOffset = 20;

Value *WaveProjection::packWorkitemId(IRBuilder<> &B, Value *X,
                                      unsigned NumDims) const {
  if (NumDims < 2)
    return X;
  Value *Y = emitWorkitemId(B, 1);
  Value *Packed =
      B.CreateOr(X,
                 B.CreateShl(Y, ConstantInt::get(I32Ty, WorkitemIdYBitOffset),
                             "tid_y_shl"),
                 "tid_xy");
  if (NumDims < 3)
    return Packed;
  Value *Z = emitWorkitemId(B, 2);
  return B.CreateOr(Packed,
                    B.CreateShl(Z,
                                ConstantInt::get(I32Ty, WorkitemIdZBitOffset),
                                "tid_z_shl"),
                    "tid_xyz");
}

Value *WaveProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                            unsigned NumDims) const {
  return packWorkitemId(B, emitWorkitemIdX(B), NumDims);
}

Value *WaveProjection::emitCurrentSourceWaveMask(IRBuilder<> &B, Value *Mask,
                                                 const Twine &Name) const {
  Type *SourceTy = sourceWaveMaskTy();
  assert(Mask->getType()->isIntegerTy() &&
         "emitCurrentSourceWaveMask expects an integer mask");
  const unsigned SourceBits = SourceTy->getPrimitiveSizeInBits();
  const unsigned MaskBits = Mask->getType()->getPrimitiveSizeInBits();
  if (Mask->getType() == SourceTy) {
    return Mask;
  }
  if (MaskBits < SourceBits) {
    return B.CreateZExtOrTrunc(Mask, SourceTy, Name);
  }

  Value *LaneId = emitLaneIdx(B);
  Value *SourceWaveBase = B.CreateAnd(
      LaneId, B.getInt32(~(static_cast<uint32_t>(sourceWaveSize()) - 1u)),
      Name + "_base");
  Value *Shift =
      B.CreateZExtOrTrunc(SourceWaveBase, Mask->getType(), Name + "_shift");
  Value *AtSourceWave = B.CreateLShr(Mask, Shift, Name + "_at_srcwave");
  return B.CreateTrunc(AtSourceWave, SourceTy, Name);
}

Value *ReplicationProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                                   unsigned NumDims) const {
  // 1-D: identical to the (already phantom-clamped) X-only seed.
  if (NumDims < 2)
    return emitWorkitemIdX(B);
  // Build from the unclamped x so the phantom-lane clamp applies once to the
  // whole packed id; otherwise a clamped-to-0 x OR'd with a non-zero Y/Z would
  // leave phantom lanes with a stray id instead of replicating lane 0.
  Value *Raw = packWorkitemId(B, WaveProjection::emitWorkitemIdX(B), NumDims);
  const unsigned SourceWaveSize = sourceWaveSize();
  const unsigned TargetWaveSize = targetWaveSize();
  if (TargetWaveSize > SourceWaveSize && MaxFlatWG > 0 &&
      MaxFlatWG < TargetWaveSize) {
    Value *Limit = ConstantInt::get(I32Ty, MaxFlatWG);
    Value *FlatLaneId = emitLaneIdx(B);
    Value *IsRealLane = B.CreateICmpULT(FlatLaneId, Limit, "tid_is_real_lane");
    // Clamp phantom upper lanes to the literal packed 0 (local id (0, 0, 0));
    // this copies nothing from lane 0. They stay hardware inactive and cannot
    // commit source-visible memory, so 0 is only an in-bounds address floor for
    // any address still computed for them.
    Raw = B.CreateSelect(IsRealLane, Raw, ConstantInt::get(I32Ty, 0),
                         "tid_phantom_clamp");
  }
  return Raw;
}

Value *WaveProjection::emitInitialExec(IRBuilder<> &B) const {
  // The architectural boot state of a dispatched wave is "every source lane
  // active", i.e. all-ones in the source-width EXEC storage. A projection that
  // decouples the modeled EXEC from the hardware EXEC overrides this hook.
  return ConstantInt::getSigned(execStorageTy(), -1);
}

Value *WaveProjection::wrapAsWWMValue(IRBuilder<> &B, Value *V,
                                      const Twine &Name) const {
  if (providesSourceWaveExecInvariant())
    return V;
  // strict.wwm is declared llvm_any_ty, but the backend lowers only integer
  // and floating-point scalars and fixed vectors thereof. Assert on that
  // subset so an unsupported type fails here rather than at lowering.
  Type *T = V->getType();
  Type *ElemTy =
      T->isVectorTy() ? cast<FixedVectorType>(T)->getElementType() : T;
  (void)ElemTy;
  assert((ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy()) &&
         "wrapAsWWMValue supports only integer / floating-point scalars "
         "and fixed-length vectors thereof");
  Module *M = B.GetInsertBlock()->getModule();
  Function *WwmFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_strict_wwm, {T});
  return B.CreateCall(WwmFn, {V}, Name);
}

// ----------------------------------------------------------------------------
// ReplicationProjection.
// ----------------------------------------------------------------------------

Value *ReplicationProjection::emitLaneActiveBit(IRBuilder<> &B,
                                                Value *ExecVal) const {
  // Project the target-lane id onto the source EXEC mask under
  // replication: target lane L is active iff bit `L mod W_src` of
  // the source EXEC mask is set. Same-wave and narrowing cases collapse
  // to the identity because `lane_id < source_wave_bits` already; the
  // modulo is a no-op and the shift happens at source width.
  //
  // Shifting at source width also sidesteps the LLVM-IR poison rule that
  // `lshr iN, M` is poison for M >= N: the pre-modulo clamps the shift
  // into [0, execBits).
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  unsigned ExecBits = ExecTy->getPrimitiveSizeInBits();
  Value *LaneIdInExec = B.CreateZExtOrTrunc(LaneId, ExecTy, "spe_lane_idx");
  // execBits is a power of two (32 or 64), so modulo is bitwise AND.
  Value *LaneMod = B.CreateAnd(
      LaneIdInExec, ConstantInt::get(ExecTy, ExecBits - 1), "spe_lane_mod");
  Value *Shifted = B.CreateLShr(ExecVal, LaneMod, "spe_exec_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(ExecTy, 1), "spe_exec_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(ExecTy, 0), "spe_lane_active");
}

const Value *
ReplicationProjection::matchBallotPredicate(const Value *Mask) const {
  using namespace PatternMatch;
  const Value *Ballot = nullptr;
  const Value *Predicate = nullptr;
  if (sourceWaveSize() == 32 && targetWaveSize() == 64 &&
      Mask->getType() == sourceWaveMaskTy() &&
      match(Mask, m_Trunc(m_Value(Ballot))) &&
      Ballot->getType() == waveMaskTy() &&
      match(Ballot, m_Intrinsic<Intrinsic::amdgcn_ballot>(m_Value(Predicate))))
    return Predicate;
  return nullptr;
}

Value *ReplicationProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                              Type *ResultTy,
                                              const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  unsigned WaveBits = waveMaskTy()->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "ballotI1ToWidth: wantedBits > waveBits (wave64 source on wave32 "
         "target) has no replication projection; this direction needs "
         "an explicit policy decision before use");
  if (WantedBits == WaveBits)
    return WaveMask;
  if (WantedBits < WaveBits)
    // Truncation is the replication projection of the target ballot
    // onto the source wave width.
    return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
  // `wantedBits > waveBits`: wave64 source on wave32 target. No correct
  // replication projection exists (the wider source wave has lanes
  // that do not exist in the narrower target), so zero-extending would
  // invent bits; fall through without returning so this direction is not
  // silently miscompiled.
}

Value *ReplicationProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                         Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *I64Ty = B.getInt64Ty();
  if (V->getType()->isPointerTy())
    V = B.CreatePtrToInt(V, I64Ty);
  Type *TargetTy = waveMaskTy();
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    // A narrow source-wave-width mask must be widened to the target
    // wave-mask width before the per-lane shift extracts a single bit.
    // Replicate the narrow mask into the upper half rather than zero-
    // extending: replication's contract is that target lane L reads
    // bit `L mod W_src` of the source mask, so a zext would make lanes in
    // the upper half always read 0.
    Value *Zext = B.CreateZExt(V, TargetTy);
    Value *Shifted = B.CreateShl(Zext, ConstantInt::get(TargetTy, SrcBits),
                                 "mask_widen_shl");
    V = B.CreateOr(Zext, Shifted, "mask_widen_replicate");
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  // Neutral `mask_*` names: this helper reads any wave mask as a per-lane
  // predicate, not only VCC, so a `vcc_` prefix would mislabel SGPR sources.
  Value *LaneIdxExt = B.CreateZExtOrTrunc(LaneIdx, TargetTy, "mask_lane_idx");
  Value *Shifted = B.CreateLShr(V, LaneIdxExt, "mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "mask_lane_i1");
}

// ----------------------------------------------------------------------------
// ReplicatedDispatchProjection.
//
// Remaps the physical workitem-id.x onto the logical source id so replica lanes
// observe the same source workitem as their primary lanes.
// ----------------------------------------------------------------------------

// logical_x = ((x_hw & ~(W_t-1)) >> log2(W_t/W_s)) | (x_hw & (W_s-1))
//
// The first term recovers the hardware wave index and rescales it to source
// lanes; the second term is the source lane within the wave (identical for a
// lane and its replica). For wave32->wave64 this is
// `((x_hw & ~63) >> 1) | (x_hw & 31)`.
static Value *emitReplicatedDispatchLogicalX(IRBuilder<> &B, Value *RawX,
                                             unsigned SourceWaveSize,
                                             unsigned TargetWaveSize) {
  assert(TargetWaveSize > SourceWaveSize &&
         (TargetWaveSize % SourceWaveSize) == 0 &&
         "replicated dispatch requires widening with an integer "
         "wave-size ratio");
  Type *Ty = RawX->getType();
  const unsigned Ratio = TargetWaveSize / SourceWaveSize;
  const unsigned RatioLog2 = llvm::Log2_32(Ratio);
  Value *WaveAligned = B.CreateAnd(
      RawX, ConstantInt::getSigned(Ty, -static_cast<int64_t>(TargetWaveSize)),
      "replicated_wave_aligned");
  Value *WaveScaled = B.CreateLShr(WaveAligned, ConstantInt::get(Ty, RatioLog2),
                                   "replicated_wave_base");
  Value *SourceLane =
      B.CreateAnd(RawX, ConstantInt::get(Ty, SourceWaveSize - 1u),
                  "replicated_source_lane");
  return B.CreateOr(WaveScaled, SourceLane, "replicated_logical_x");
}

Value *ReplicatedDispatchProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  return emitWorkitemId(B, 0);
}

Value *ReplicatedDispatchProjection::emitWorkitemId(IRBuilder<> &B,
                                                    unsigned Dim) const {
  assert(Dim < 3 && "invalid workitem dimension");
  if (Launch.Mapping == KernelLaunchRequirements::Kind::Replicated1D &&
      Dim != 0)
    return B.getInt32(0);
  Value *Raw = emitTargetWorkitemId(B, 0);
  Value *Flat = emitReplicatedDispatchLogicalX(B, Raw, sourceWaveSize(),
                                               targetWaveSize());
  if (Launch.Mapping == KernelLaunchRequirements::Kind::Replicated1D)
    return Flat;

  Value *SizeX = emitWorkgroupSize(B, 0);
  if (Dim == 0)
    return B.CreateURem(Flat, SizeX, "logical_x");
  Value *Row = B.CreateUDiv(Flat, SizeX);
  Value *SizeY = emitWorkgroupSize(B, 1);
  return Dim == 1 ? B.CreateURem(Row, SizeY, "logical_y")
                  : B.CreateUDiv(Row, SizeY, "logical_z");
}

Value *WaveProjection::emitWorkgroupSize(IRBuilder<> &B, unsigned Dim) const {
  Value *Dispatch = B.CreateCall(Intrinsic::getOrInsertDeclaration(
      B.GetInsertBlock()->getModule(), Intrinsic::amdgcn_dispatch_ptr));
  Value *Address = B.CreateConstGEP1_64(
      B.getInt8Ty(), Dispatch,
      DispatchWorkgroupSizeXOffset + DispatchWorkgroupSizeFieldBytes * Dim);
  Value *Size = B.CreateAlignedLoad(B.getInt16Ty(), Address, Align(2));
  return B.CreateZExt(Size, B.getInt32Ty());
}

Value *ReplicatedDispatchProjection::emitWorkgroupSize(IRBuilder<> &B,
                                                       unsigned Dim) const {
  if (Launch.RequiredWorkgroupSize)
    return B.getInt32((*Launch.RequiredWorkgroupSize)[Dim]);
  if (!Launch.WorkgroupSizeArgOffsets) {
    assert(Launch.Mapping == KernelLaunchRequirements::Kind::Replicated1D &&
           Launch.RequiresWholeSourceWaves && "missing logical group sizes");
    if (Dim != 0)
      return B.getInt32(1);
    Value *Physical = WaveProjection::emitWorkgroupSize(B, 0);
    return B.CreateUDiv(Physical, B.getInt32(replicationFactor()));
  }
  Value *Kernarg = B.CreateCall(Intrinsic::getOrInsertDeclaration(
      B.GetInsertBlock()->getModule(), Intrinsic::amdgcn_kernarg_segment_ptr));
  Value *Address = B.CreateConstGEP1_64(B.getInt8Ty(), Kernarg,
                                        (*Launch.WorkgroupSizeArgOffsets)[Dim]);
  Value *Size = B.CreateAlignedLoad(B.getInt16Ty(), Address, Align(1));
  return B.CreateZExt(Size, B.getInt32Ty());
}

Value *ReplicatedDispatchProjection::emitInitialExec(IRBuilder<> &B) const {
  if (Launch.RequiresWholeSourceWaves)
    return B.getInt32(-1);
  Value *SizeX = emitWorkgroupSize(B, 0);
  Value *SizeY = emitWorkgroupSize(B, 1);
  Value *SizeZ = emitWorkgroupSize(B, 2);
  Value *RowWorkitems = B.CreateMul(SizeX, SizeY);
  Value *Workitems = B.CreateMul(RowWorkitems, SizeZ);
  Value *Raw = emitTargetWorkitemId(B, 0);
  Value *Flat = emitReplicatedDispatchLogicalX(B, Raw, sourceWaveSize(),
                                               targetWaveSize());
  Value *Active = B.CreateICmpULT(Flat, Workitems, "source_entry_active");
  return ballotI1ToWidth(B, Active, sourceWaveMaskTy(), "source_entry_exec");
}

ReplicatedDispatchProjection::ReplicatedDispatchProjection(
    const MCSubtargetInfo &Source, const MCSubtargetInfo &Target, Type *I32Ty,
    Type *I64Ty, const KernelLaunchRequirements &Launch)
    : ReplicationProjection(Source, Target, I32Ty, I64Ty), Launch(Launch) {
  assert(targetWaveSize() > sourceWaveSize() &&
         targetWaveSize() % sourceWaveSize() == 0 &&
         "replicated dispatch requires widening by an integer factor");
  ReplicationFactor = targetWaveSize() / sourceWaveSize();
  ProvidesSourceWaveExecInvariant = true;
}

Value *ReplicatedDispatchProjection::emitSourceWaveId(IRBuilder<> &B) const {
  Value *X = emitTargetWorkitemId(B, 0);
  Value *Wave = B.CreateUDiv(X, B.getInt32(targetWaveSize()));
  return B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {B.getInt32Ty()},
                           {Wave}, nullptr, "source_wave_id");
}

Value *
ReplicatedDispatchProjection::emitPackedWorkitemId(IRBuilder<> &B,
                                                   unsigned NumDims) const {
  return packWorkitemId(B, emitWorkitemIdX(B), NumDims);
}

// The target ballot is i64; source EXEC and scalar masks use an independent
// i32 value per source wave. Selecting the corresponding half before
// narrowing preserves both masks through scalar operations and SGPR saves.
WaveNativeProjection::WaveNativeProjection(const MCSubtargetInfo &Source,
                                           const MCSubtargetInfo &Target,
                                           Type *I32Ty, Type *I64Ty,
                                           bool AllowDivergentScalarControlFlow)
    : WaveProjection(Source, Target, I32Ty, I64Ty) {
  assert(sourceWaveSize() == 32 && targetWaveSize() == 64 &&
         "WaveNativeProjection requires wave32 to wave64");
  NumSourceWavesPerTarget = 2;
  AllowsDivergentScalarControlFlow = AllowDivergentScalarControlFlow;
  ProvidesSourceWaveExecInvariant = true;
  PreservesMbcntDerivedExec = true;
}

Value *WaveNativeProjection::emitSourceWaveId(IRBuilder<> &B) const {
  Value *TargetWave = emitTargetWaveId(B);
  Value *FirstSourceWave = B.CreateMul(
      TargetWave, B.getInt32(numSourceWavesPerTarget()), "first_source_wave");
  Value *SourceWaveInTarget = B.CreateUDiv(
      emitLaneIdx(B), B.getInt32(sourceWaveSize()), "source_wave_in_target");
  return B.CreateAdd(FirstSourceWave, SourceWaveInTarget, "source_wave_id");
}

Value *WaveNativeProjection::emitInitialExec(IRBuilder<> &B) const {
  // Enable whole-wave execution for source scalar instructions and lane
  // collectives. The intrinsic returns each lane's original activity; retain
  // it so partial waves cannot acquire additional vector memory effects.
  EntryActive = B.CreateIntrinsic(Intrinsic::amdgcn_init_whole_wave, {}, {},
                                  nullptr, "orig_active");
  return ballotI1ToWidth(B, EntryActive, sourceWaveMaskTy(), "saved_exec");
}

Value *WaveNativeProjection::emitLaneActiveBit(IRBuilder<> &B,
                                               Value *ExecVal) const {
  assert(ExecVal->getType() == sourceWaveMaskTy() &&
         "EXEC must have source wave width");
  assert(EntryActive && "initial EXEC must be emitted first");
  Value *Active = extractLaneBitFromWaveMask(B, ExecVal);
  return B.CreateSelect(EntryActive, Active, B.getFalse(), "dispatched_active");
}

const Value *
WaveNativeProjection::matchBallotPredicate(const Value *Mask) const {
  using namespace PatternMatch;
  const Value *Ballot = nullptr;
  const Value *Predicate = nullptr;
  if (CachedLaneIdx && Mask->getType() == sourceWaveMaskTy() &&
      match(Mask, m_Trunc(m_LShr(m_Value(Ballot),
                                 m_ZExt(m_And(m_Specific(CachedLaneIdx),
                                              m_SpecificInt(~(sourceWaveSize() -
                                                              1u))))))) &&
      Ballot->getType() == waveMaskTy() &&
      match(Ballot, m_Intrinsic<Intrinsic::amdgcn_ballot>(m_Value(Predicate))))
    return Predicate;
  return nullptr;
}

Value *WaveNativeProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                             Type *ResultTy,
                                             const Twine &Name) const {
  assert(Pred->getType()->isIntegerTy(1) && "expected a predicate");
  Value *Mask = B.CreateIntrinsic(Intrinsic::amdgcn_ballot, {waveMaskTy()},
                                  {Pred}, nullptr, Name);
  if (ResultTy == waveMaskTy())
    return Mask;
  // SGPR values may differ between the two source waves. Select the slice
  // before narrowing so ordinary scalar operations preserve both masks.
  Mask = emitCurrentSourceWaveMask(B, Mask, Name + "_source");
  return B.CreateZExtOrTrunc(Mask, ResultTy, Name);
}

Value *WaveNativeProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                        Value *V) const {
  if (V->getType()->isIntegerTy(1))
    return V;
  V = B.CreateZExtOrTrunc(V, sourceWaveMaskTy());
  Value *Lane = B.CreateAnd(emitLaneIdx(B), B.getInt32(31), "source_lane");
  Value *Bit = B.CreateAnd(B.CreateLShr(V, Lane), B.getInt32(1), "mask_bit");
  return B.CreateICmpNE(Bit, B.getInt32(0), "lane_active");
}

// ----------------------------------------------------------------------------
// ThreadLoopProjection.
// ----------------------------------------------------------------------------

ThreadLoopProjection::ThreadLoopProjection(const MCSubtargetInfo &Source,
                                           const MCSubtargetInfo &Target,
                                           Type *I32Ty, Type *I64Ty)
    : WaveProjection(Source, Target, I32Ty, I64Ty) {
  const unsigned SourceWaveSize = sourceWaveSize();
  const unsigned TargetWaveSize = targetWaveSize();
  assert(TargetWaveSize > SourceWaveSize &&
         "ThreadLoopProjection is defined only for widening "
         "(target wave > source wave)");

  assert((TargetWaveSize % SourceWaveSize) == 0 &&
         "ThreadLoopProjection requires target wave size to be an integer "
         "multiple of source wave size");

  ExecStorageTy = waveMaskTy();
  NumSourceWavesPerTarget = TargetWaveSize / SourceWaveSize;
  SourceWaveScopedLaneOps = true;
}

Value *ThreadLoopProjection::emitWorkitemIdX(IRBuilder<> &B) const {
  assert(IterationAlloca &&
         "ThreadLoopProjection::emitWorkitemIdX requires an iteration alloca; "
         "raiser must call setIterationAlloca before emitting source workitem "
         "ids");
  Value *Tid = emitTargetWorkitemId(B, 0);
  Value *LaneId = emitLaneIdx(B);
  const unsigned SrcBits = sourceWaveSize();
  const unsigned TgtBits = targetWaveSize();
  Value *Iter = B.CreateLoad(B.getInt32Ty(), IterationAlloca, "tl_iter");
  Value *Base = B.CreateAnd(Tid, B.getInt32(~(TgtBits - 1u)), "tl_tid_base");
  Value *SourceLane =
      B.CreateAnd(LaneId, B.getInt32(SrcBits - 1u), "tl_source_lane");
  Value *WaveOffset =
      B.CreateMul(Iter, B.getInt32(SrcBits), "tl_source_wave_off");
  return B.CreateAdd(B.CreateAdd(Base, WaveOffset, "tl_tid_wave_base"),
                     SourceLane, "tl_tid");
}

Value *ThreadLoopProjection::emitSourceWaveId(IRBuilder<> &B) const {
  assert(IterationAlloca &&
         "ThreadLoopProjection::emitSourceWaveId requires an iteration alloca");
  Value *TargetWave = emitTargetWaveId(B);
  Value *FirstSourceWave = B.CreateMul(
      TargetWave, B.getInt32(numSourceWavesPerTarget()), "first_source_wave");
  Value *Iteration =
      B.CreateLoad(B.getInt32Ty(), IterationAlloca, "source_wave_in_target");
  return B.CreateAdd(FirstSourceWave, Iteration, "source_wave_id");
}

Value *ThreadLoopProjection::emitLaneActiveBit(IRBuilder<> &B,
                                               Value *ExecVal) const {
  Value *LaneId = emitLaneIdx(B);
  Type *ExecTy = ExecVal->getType();
  const unsigned SourceBits = sourceWaveMaskTy()->getPrimitiveSizeInBits();
  Value *LaneIdInExec = B.CreateZExtOrTrunc(LaneId, ExecTy, "tl_lane_idx");
  Value *LaneMod = B.CreateAnd(
      LaneIdInExec, ConstantInt::get(ExecTy, SourceBits - 1), "tl_lane_mod");
  Value *Shifted = B.CreateLShr(ExecVal, LaneMod, "tl_exec_at_lane");
  Value *Bit = B.CreateAnd(Shifted, ConstantInt::get(ExecTy, 1), "tl_exec_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(ExecTy, 0), "tl_lane_active");
}

Value *ThreadLoopProjection::ballotI1ToWidth(IRBuilder<> &B, Value *Pred,
                                             Type *ResultTy,
                                             const Twine &Name) const {
  assert(Pred->getType() == B.getInt1Ty() &&
         "ballotI1ToWidth requires an i1 predicate");
  Module *M = B.GetInsertBlock()->getModule();
  Function *Ballot = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ballot, {waveMaskTy()});
  Value *WaveMask = B.CreateCall(Ballot, {Pred}, Name);
  const unsigned WantedBits = ResultTy->getPrimitiveSizeInBits();
  const unsigned WaveBits = waveMaskTy()->getPrimitiveSizeInBits();
  assert(WantedBits <= WaveBits &&
         "ThreadLoopProjection::ballotI1ToWidth requires resultTy <= target "
         "wave mask width");
  if (WantedBits == WaveBits)
    return WaveMask;
  return B.CreateTrunc(WaveMask, ResultTy, Name + "_trunc");
}

Value *ThreadLoopProjection::extractLaneBitFromWaveMask(IRBuilder<> &B,
                                                        Value *V) const {
  if (V->getType() == B.getInt1Ty())
    return V;
  Type *TargetTy = V->getType()->getPrimitiveSizeInBits() >
                           sourceWaveMaskTy()->getPrimitiveSizeInBits()
                       ? waveMaskTy()
                       : sourceWaveMaskTy();
  unsigned SrcBits = V->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = TargetTy->getPrimitiveSizeInBits();
  if (SrcBits < DstBits) {
    V = B.CreateZExt(V, TargetTy);
  } else if (SrcBits > DstBits) {
    V = B.CreateTrunc(V, TargetTy);
  } else if (V->getType() != TargetTy) {
    V = B.CreateBitCast(V, TargetTy);
  }
  Value *LaneIdx = emitLaneIdx(B);
  Value *LaneIdxExt =
      B.CreateZExtOrTrunc(LaneIdx, TargetTy, "tl_mask_lane_idx");
  Value *ShiftIdx =
      (TargetTy == waveMaskTy())
          ? LaneIdxExt
          : B.CreateAnd(LaneIdxExt, ConstantInt::get(TargetTy, DstBits - 1),
                        "tl_mask_lane_mod");
  Value *Shifted = B.CreateLShr(V, ShiftIdx, "tl_mask_at_lane");
  Value *Bit =
      B.CreateAnd(Shifted, ConstantInt::get(TargetTy, 1), "tl_mask_lane_bit");
  return B.CreateICmpNE(Bit, ConstantInt::get(TargetTy, 0), "tl_mask_lane_i1");
}

// ----------------------------------------------------------------------------
// EXEC-writer detection.
// ----------------------------------------------------------------------------

bool instructionWritesEXEC(const DecodedInst &Di, const MCState &Mc) {
  if (Di.defsExec())
    return true;
  // On a wave32 source, hardware EXEC is 32-bit (== EXEC_LO) and EXEC_HI is a
  // free scratch scalar, so an explicit def of EXEC_HI alone is a scratch
  // write, not an EXEC write.
  const bool SourceIsWave32 = getWaveSize(*Mc.SubtargetInfo) == 32;
  const MCInstrDesc &Desc = Mc.InstrInfo->get(Di.Inst.getOpcode());
  for (unsigned I = 0; I < Desc.getNumDefs() && I < Di.Inst.getNumOperands();
       ++I) {
    const MCOperand &Mop = Di.Inst.getOperand(I);
    if (!Mop.isReg() || !Mop.getReg())
      continue;
    // EXEC has no subtarget-specific MC aliases, so the raw MC register already
    // matches the pseudo-register id; no mc2PseudoReg normalization is needed.
    MCRegister Reg = Mop.getReg();
    if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO)
      return true;
    if (Reg == AMDGPU::EXEC_HI && !SourceIsWave32)
      return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Cross-wave warning.
// ----------------------------------------------------------------------------

bool emitCrossWaveWarning(const WaveProjection &Proj, const MCState &Mc,
                          ArrayRef<DecodedInst> Insts, StringRef SourceIsa,
                          StringRef TargetIsa) {
  const unsigned SourceWaveSize = Proj.sourceWaveSize();
  const unsigned TargetWaveSize = Proj.targetWaveSize();
  if (SourceWaveSize == TargetWaveSize)
    return false;

  const DecodedInst *FirstExecWriter = nullptr;
  for (const DecodedInst &Di : Insts) {
    if (instructionWritesEXEC(Di, Mc)) {
      FirstExecWriter = &Di;
      break;
    }
  }
  if (!FirstExecWriter)
    return false;

  // Emit a warn-only diagnostic under -debug-only=wave-projection.
  LLVM_DEBUG({
    dbgs() << "transpiler: WARNING: cross-wave translation of an "
              "EXEC-manipulating kernel relies on replication, "
              "which is not provably correct in general.\n"
           << "  source ISA wave size: " << SourceWaveSize << " (" << SourceIsa
           << ")\n"
           << "  target ISA wave size: " << TargetWaveSize << " ("
           << (TargetIsa.empty() ? SourceIsa : TargetIsa) << ")\n"
           << "  first EXEC-writer: " << getMnemonic(Mc, FirstExecWriter->Inst)
           << " at offset 0x"
           << format_hex_no_prefix(FirstExecWriter->Offset, 4) << "\n"
           << "  rationale: the kernel manipulates EXEC; replicating it "
              "across wave halves will double per-lane side effects in a "
              "way the source author did not specify. Empirically this is "
              "correct for kernels whose EXEC writers are lane-position-"
              "independent (pointwise ops with bounds checks against a "
              "uniform >= target_wave_bits).\n";
  });
  return true;
}

} // namespace COMGR::transpiler
