//===- register-state.h - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_REGISTER_STATE_H
#define TRANSPILER_REGISTER_STATE_H

#include "transpiler/common/kernel-meta.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/reg-file.h"
#include "transpiler/raiser/user-sgpr-layout.h"
#include "transpiler/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace llvm {
class DominatorTree;
}

namespace COMGR::transpiler {

// The source architectural registers as the raiser sees them. Scalar wave-mask
// results also keep a per-lane shadow, keyed by the SGPR range they occupy, so
// widening projections do not lose independently computed source-wave masks.
class RegisterState {
public:
  // Build the register state for the source kernel described by Meta, with the
  // SGPRs the source ABI preloads before entry already seeded. B must be
  // positioned in the entry block: the register file, the cross-block shadow
  // storage, and the seeds are emitted there. Fails when the kernel descriptor
  // and the metadata disagree on the user-SGPR layout, and when the layout
  // preloads an entry source the target cannot reproduce.
  static llvm::Expected<RegisterState> create(llvm::IRBuilder<> &B,
                                              const WaveProjection &Projection,
                                              const MCState &MC,
                                              const KernelMeta &Meta);

  // Alloca-backed storage for the source architectural registers.
  AllocaRegFile &regFile() { return Regs; }
  // What each SGPR holds at source kernel entry.
  const UserSgprLayout &layout() const { return Layout; }

  /// Reconstruct supported dispatch fields and prove other entry state unused.
  /// All instruction and projection checks must run before this finalization.
  llvm::Error validateEntrySgprs();

  // Active low byte of S_SET_VGPR_MSB. Each two-bit field selects the high
  // VGPR bank for a format-defined operand slot.
  uint8_t vgprMsBs() { return blockState().VgprMsBs; }
  void setVgprMsBs(uint8_t Value) { blockState().VgprMsBs = Value; }

  // VGPR index adjustments for the instruction passed to computeVGPRAdjust,
  // indexed by MC operand index.
  llvm::ArrayRef<unsigned> currentVgprAdjust() const {
    return CurrentVgprAdjust;
  }

  // Compute VGPR bank adjustments for the instruction's format-defined slots.
  void computeVGPRAdjust(const DecodedInst &Di);

  // Resolve the register operand at OperandIndex to the source register it
  // names. Fails on a register this raiser does not model.
  llvm::Expected<ParsedReg> parseReg(const DecodedInst &Di,
                                     unsigned OperandIndex) const;

  // Read the operand at OpIdx as a 32-bit value, resolving registers through
  // the reg-file and immediates through the MC operand.
  llvm::Expected<llvm::Value *> readOp32(const DecodedInst &Di, unsigned OpIdx);
  // Read the operand at OpIdx as a 64-bit value, pairing adjacent registers.
  llvm::Expected<llvm::Value *> readOp64(const DecodedInst &Di, unsigned OpIdx);
  // Store an SCC value known not to depend on a source-image address.
  void storeSCC(llvm::IRBuilder<> &B, llvm::Value *V);

  // Store an SCC value computed from a source-image address. Reads of this
  // value are refused because relocation may change it.
  void storeSourceImageSCC(llvm::IRBuilder<> &B, llvm::Value *V);

  // Read SCC, refusing it when its value may depend on a source-image address.
  llvm::Expected<llvm::Value *> readSCC(const DecodedInst &Di);

  // Return a block-local SCC constant independent of source relocation.
  std::optional<bool> knownSCC() { return blockState().SccConstant; }

  // Read the SGPR at Idx, or the pair based there, naming the register by
  // index rather than by an operand. Di is the instruction the read serves,
  // which a refusal names.
  llvm::Expected<llvm::Value *> readSgpr32(const DecodedInst &Di, unsigned Idx);
  llvm::Expected<llvm::Value *> readSgpr64(const DecodedInst &Di, unsigned Idx);
  // Number of SGPRs backed by the register file.
  unsigned numSgprs() const { return static_cast<unsigned>(Regs.Sgpr.size()); }

  // Read M0, which the message opcodes take their payload from without
  // naming it in an operand.
  llvm::Value *readM0() {
    return Regs.readReg32(B, ParsedReg{ParsedReg::M0, 0, 1});
  }

  // Read a mask at target EXEC width, replicating narrower source-wave bits.
  llvm::Expected<llvm::Value *> readOpExecWidth(const DecodedInst &Di,
                                                unsigned OpIdx);
  // Read an operand's per-lane wave-mask value, or null when no shadow matches
  // the operand's full register width.
  llvm::Expected<llvm::Value *> readOpWaveMaskI1(const DecodedInst &Di,
                                                 unsigned OpIdx);
  // Read the mask a source-wave instruction should see, e.g. for `v_mbcnt_lo`.
  // EXEC/VCC/SGPR-shadow masks are projected; scalars use readOp32.
  llvm::Expected<llvm::Value *> readOpSourceWaveMask32(const DecodedInst &Di,
                                                       unsigned OpIdx);

  // Return whether the current target lane is active in the logical EXEC mask.
  // The result is cached for one source instruction and invalidated by EXEC
  // writes.
  llvm::Value *emitLaneActiveBit();

  // Invalidate cached lane activity after an EXEC write or instruction
  // boundary.
  void resetLaneActiveCache() { blockState().CachedLaneActive = nullptr; }

  // Read EXEC at the width the projection stores it in.
  llvm::Value *readExec() { return Regs.readExecWidth(B); }

  // Store EXEC and invalidate cached lane activity.
  void storeExec(llvm::Value *V) {
    Regs.storeExec(B, V);
    resetLaneActiveCache();
  }

  // Write V to the register Pr names, at the register's width. VGPR and AGPR
  // writes are predicated on EXEC; scalar writes commit for the whole wave.
  void writeReg32(ParsedReg Pr, llvm::Value *V);
  void writeReg64(ParsedReg Pr, llvm::Value *V);
  // Write a value spanning more than two dwords, such as an image descriptor.
  void writeRegVec(ParsedReg Pr, llvm::Value *V);
  // Write a wave mask at the target EXEC width.
  void writeRegExecWidth(ParsedReg Pr, llvm::Value *V);

  // Emit Body in a lane-active region and write the value it returns to the
  // per-lane register Pr names. For a value an inactive lane must not compute
  // at all, such as one loaded from a per-lane address.
  void writeReg32UnderExec(ParsedReg Pr,
                           llvm::function_ref<llvm::Value *()> Body);

  // Write V to a vector register by resolved index, predicated on EXEC.
  void storeVGPR32(unsigned Idx, llvm::Value *V);
  void storeVGPR64(unsigned Idx, llvm::Value *V);
  void storeAGPR32(unsigned Idx, llvm::Value *V);

  // Emit Body under a branch on the lane being active, and leave the builder
  // in the block the taken and untaken paths join at. This preserves inactive
  // lanes for per-lane side effects.
  void emitUnderExec(llvm::function_ref<void()> Body);

  /// Under replication, emit a memory effect only in primary lanes.
  /// The caller supplies the instruction's EXEC and bounds predication;
  /// Body must not update architectural registers.
  void emitMemoryEffect(llvm::function_ref<void()> Body);

  /// Under replication, read in primary lanes and broadcast results to
  /// replicas. Scalar reads execute in lane zero, independently of source EXEC.
  /// The caller supplies the instruction's EXEC and bounds predication and
  /// writes back the returned value.
  llvm::Value *emitMemoryValue(llvm::function_ref<llvm::Value *()> Body,
                               bool IsScalar = false);

  // Emit Body when the source wave has any active lane in EXEC.
  void emitWithNonzeroExec(llvm::function_ref<void()> Body);

  // Record CmpI1 as the per-lane compare a V_CMP wrote to SGPR BaseIdx, both
  // for reuse within the block and in the cross-block shadow storage. IsPair
  // says whether the destination spans BaseIdx and its successor.
  void recordSgprWaveMaskI1(unsigned BaseIdx, llvm::Value *CmpI1, bool IsPair);

  // Record MaskI1 as the wave-mask value written to Dst.
  void recordWaveMaskI1(ParsedReg Dst, llvm::Value *MaskI1);

  // Emit a test of whether the source wave holding the current target lane has
  // any lane active in EXEC.
  llvm::Value *emitCurrentSourceWaveHasActiveLane();

  // Record V as the value SGPR pair BaseIdx holds for the source wave holding
  // the current target lane, leaving the value recorded for other source waves
  // in place. Does nothing unless the projection packs whole source waves.
  void recordSourceWaveSgprPair(unsigned BaseIdx, llvm::Value *V);

  // Emit a read of the value recorded for SGPR pair BaseIdx, or Fallback when
  // nothing was recorded.
  llvm::Value *materializeSourceWaveSgprPair(unsigned BaseIdx,
                                             llvm::Value *Fallback);

  // Return the compare recorded for this SGPR range in this block, or null when
  // unavailable.
  llvm::Value *lookupSgprWaveMaskI1(unsigned BaseIdx, bool Is64) {
    auto &Recorded = blockState().LastSgprWaveMaskI1;
    auto It = Recorded.find(BaseIdx);
    if (It == Recorded.end())
      return nullptr;
    return It->second.IsPair == Is64 ? It->second.I1 : nullptr;
  }

  // Emit a test of whether the source wave holding the current target lane has
  // no lane set in EXEC, and the same for VCC. These are the wave-level
  // conditions `execz` and `vccz` name.
  llvm::Value *emitExecIsZero();
  llvm::Value *emitVccIsZero();

  // Emit a read of the wave mask shadowed for SGPR BaseIdx, and of the bit
  // saying whether that shadow is valid. Both return null for an SGPR with no
  // shadow storage.
  llvm::Value *loadSgprWaveMaskExec(unsigned BaseIdx) const;
  llvm::Value *loadSgprWaveMaskValid(unsigned BaseIdx) const;

  // Invalidate the facts a write to SGPR BaseIdx invalidates. A pair is keyed
  // by its low SGPR, so writing its high SGPR also invalidates the preceding
  // entry. Single-SGPR entries remain independent.
  void invalidateSgprWaveMaskI1(unsigned BaseIdx);

  // Record the source-image address held by an SGPR pair.
  void recordSourceImageSgprPairAddr(unsigned BaseIdx, uint64_t Value);

  // Return the source-image address known for this pair in the current block.
  std::optional<uint64_t> lookupSourceImageSgprPairAddr(unsigned BaseIdx);

  // Record that the low half of a split source code-object address
  // displacement left carry Carry in SCC for SGPR pair BaseIdx, and that the
  // high-half add consuming it starts at source offset NextOffset. That offset
  // is the one directly after the low add, so any instruction placed between
  // the two takes it and the carry goes unclaimed rather than being read as
  // still belonging to the pair.
  void recordSourceImageCarry(unsigned BaseIdx, bool Carry,
                              uint64_t NextOffset) {
    blockState().SourceImageCarry =
        BlockState::SourceImageCarryState{BaseIdx, Carry, NextOffset};
  }

  // Consume the carry recorded for SGPR pair BaseIdx on behalf of a high-half
  // add starting at source offset Offset. Return no value when no such carry
  // is waiting.
  std::optional<bool> takeSourceImageCarry(unsigned BaseIdx, uint64_t Offset);

  // Reject deferred reads that may observe source-address bits on any reachable
  // path. Requires SSA promotion of the register and provenance allocas.
  llvm::Error refuseDeferredSourceImageReads(const llvm::DominatorTree &DT);

  // Track the value written to M0, which the relative-addressing opcodes need
  // as a constant to resolve the register index they name. A non-constant
  // write, and any block boundary, gives up the constant.
  void updateM0Const(llvm::Value *V);
  std::optional<uint64_t> getM0Const() { return blockState().M0Const; }

  // True while TTMP8 still holds its source kernel-entry value.
  bool isTTMP8EntryValueAvailable() const { return TTMP8EntryValueAvailable; }

  // Emit stores marking every cross-block SGPR shadow invalid.
  void invalidateSgprShadows();

  // Append every alloca backing the register state, register file included, to
  // Out for SSA promotion.
  void collectAllocas(llvm::SmallVectorImpl<llvm::AllocaInst *> &Out) const;

private:
  llvm::SmallVector<llvm::WeakTrackingVH> UnavailableEntryValues;

  // An operand read whose source-address provenance needs a CFG check.
  struct DeferredSourceImageRead {
    const DecodedInst *Di;
    // No index denotes SCC.
    std::optional<unsigned> Idx;
    llvm::WeakTrackingVH Provenance;
    llvm::BasicBlock *Block;
  };

  // Diagnose a source-address read of an SGPR, or SCC when no index is given.
  llvm::Error refuseSourceImageRead(const DecodedInst &Di,
                                    std::optional<unsigned> Idx);

  // Decide a read of the registers named either by a base index and a width in
  // dwords or by the operand that reads them, refusing it when one of them
  // holds a source code-object address and deferring it when the raise cannot
  // yet tell.
  llvm::Error refuseSourceImageRead(const DecodedInst &Di, unsigned BaseIdx,
                                    unsigned WidthInDwords);
  llvm::Error refuseSourceImageRead(const DecodedInst &Di, const ParsedReg &Pr);

  // Emit a conditional region while preserving register-state tracking.
  void emitUnderCondition(llvm::Value *Condition,
                          llvm::function_ref<void()> Body);

  RegisterState(llvm::IRBuilder<> &B, const WaveProjection &Projection,
                const MCState &MC, UserSgprLayout Layout);

  // Give the preloaded entry SGPRs the values the source ABI hands them.
  llvm::Error seedEntrySgprs(const KernelMeta &Meta);

  // Give the preloaded entry VGPRs the values the source ABI hands them.
  void seedEntryVgprs(const KernelMeta &Meta);

  // Storage shadowing one SGPR across block boundaries.
  struct SgprShadow {
    // Wave mask last written to this SGPR, at target EXEC width.
    llvm::AllocaInst *WaveMask;
    // Whether WaveMask still describes what the SGPR holds.
    llvm::AllocaInst *WaveMaskValid;
    // Whether the mask spans this SGPR and its successor.
    llvm::AllocaInst *WaveMaskIsPair;
    // Value this SGPR pair holds for the current source wave.
    llvm::AllocaInst *SourceWavePair;
    // Whether SourceWavePair was ever written for this source wave.
    llvm::AllocaInst *SourceWavePairValid;
    // Whether this dword holds part of a source-image address.
    llvm::AllocaInst *SourceImage;
  };

  // A V_CMP result kept for reuse while the SGPR it wrote remains valid.
  struct WaveMaskEntry {
    llvm::Value *I1 = nullptr;
    // Whether the destination spans this SGPR and its successor.
    bool IsPair = false;
  };

  // What holds only while one source block is being raised. These rest on
  // values that dominate only from inside that block, or on architectural
  // state a block does not inherit from its predecessors.
  struct BlockState {
    // Lane-active bit reused until an EXEC write or instruction boundary.
    llvm::Value *CachedLaneActive = nullptr;
    // Compares keyed by the SGPR they were written to.
    llvm::DenseMap<unsigned, WaveMaskEntry> LastSgprWaveMaskI1;
    // Source-image addresses proven for PC-relative literal loads.
    llvm::DenseMap<unsigned, uint64_t> SourceImageSgprPairAddrShadow;
    // Carry that the low half of a split source code-object address
    // displacement left in SCC: the SGPR pair it displaces, the carry itself,
    // and the source offset at which the high-half add that consumes it
    // starts.
    struct SourceImageCarryState {
      unsigned PairBaseIdx;
      bool Carry;
      uint64_t NextOffset;
    };
    std::optional<SourceImageCarryState> SourceImageCarry;
    std::optional<bool> SccConstant;
    bool SccFromSourceImage = false;
    llvm::DenseSet<unsigned> SourceImageDwords;
    // Constant value last stored to M0.
    std::optional<uint64_t> M0Const;
    // Active low byte of S_SET_VGPR_MSB. Architectural rather than raise-time:
    // LLVM's VGPR-encoding lowering resets the mode at every block boundary,
    // so a raised block must not inherit what a predecessor left set.
    uint8_t VgprMsBs = 0;
  };

  // The block-local state, dropped when the builder has moved to a block other
  // than the one that established it. Every accessor of that state goes
  // through here, which is what keeps the boundary from depending on a caller
  // remembering to announce it.
  BlockState &blockState();

  // Declare the block the builder now sits in a continuation of the one the
  // state belongs to. Only sound for a block that one dominates.
  void carryStateIntoCurrentBlock() { StateBlock = B.GetInsertBlock(); }

  // Builder the register accesses are emitted into. Its insertion point moves
  // as raising progresses.
  llvm::IRBuilder<> &B;
  // Translation between the source and target wave sizes.
  const WaveProjection &Projection;
  // MC layer for the source ISA, shared by every kernel in the code object.
  const MCState &MC;

  // Source architectural registers, allocated in the entry block.
  AllocaRegFile Regs;
  // What each SGPR holds at source kernel entry.
  UserSgprLayout Layout;

  // VGPR bank adjustment per MC operand of the instruction being raised.
  llvm::SmallVector<unsigned> CurrentVgprAdjust;

  bool TTMP8EntryValueAvailable = true;

  // What holds within one source block, and the block it holds in.
  BlockState State;
  llvm::BasicBlock *StateBlock = nullptr;

  // Shadow storage per SGPR. Cross-block values live in allocas to avoid
  // carrying SSA values that do not dominate their uses.
  llvm::SmallVector<SgprShadow> SgprShadows;

  llvm::SmallVector<DeferredSourceImageRead> DeferredSourceImageReads;
  llvm::AllocaInst *SourceImageScc;
};

} // namespace COMGR::transpiler

#endif
