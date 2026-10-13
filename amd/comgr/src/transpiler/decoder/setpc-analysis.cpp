//===- setpc-analysis.cpp - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "setpc-analysis.h"

#include "decode.h"
#include "mc-state.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

// Index of the SGPR that `Reg` names, or nullopt when it is not one of the
// general scalar registers the analysis follows. A pair is named by its low
// register, which is the index this returns for either width.
std::optional<unsigned> sgprIndex(const MCRegisterInfo &MRI, MCRegister Reg) {
  if (!Reg)
    return std::nullopt;
  MCRegister Low = MRI.getSubReg(Reg, AMDGPU::sub0);
  Low = stripRegEncoding(Low ? Low : Reg);

  if (!MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Low))
    return std::nullopt;
  // That class also holds the two halves of the condition register, which
  // cannot carry a captured program counter.
  if (Low == AMDGPU::VCC_LO || Low == AMDGPU::VCC_HI)
    return std::nullopt;
  return MRI.getEncodingValue(Low) & AMDGPU::HWEncoding::REG_IDX_MASK;
}

// Most source offsets one transfer may reach before the analysis stops
// enumerating and refuses it. A dispatcher a compiler wrote stays far below
// this, so a wider fan-out is more likely a value the analysis got wrong than
// a table it can state.
constexpr unsigned MaxSetPcTargets = 16;

// The source offsets one register pair may hold, as a dataflow lattice.
//
// The bottom element is the empty offset set: nothing reaches the pair. A join
// unions the offset sets, and a value only ever moves up. The two top elements
// carry no offsets. Both refuse the transfers that read them, and differ only
// in the refusal they produce.
class TargetOffsets {
public:
  TargetOffsets() = default;

  static TargetOffsets single(uint64_t Offset, bool Incomplete = false) {
    TargetOffsets Result;
    Result.Offsets.push_back(Offset);
    Result.Incomplete = Incomplete;
    return Result;
  }

  // Top: some path leaves the pair holding a value the analysis cannot name.
  static TargetOffsets unnameable() {
    TargetOffsets Result;
    Result.Top = Reason::Unnameable;
    return Result;
  }

  bool isUnnameable() const { return Top == Reason::Unnameable; }
  bool isTooMany() const { return Top == Reason::TooMany; }
  bool isTop() const { return Top != Reason::None; }
  bool isIncomplete() const { return Incomplete; }

  // The offsets that reach the pair, ascending and distinct. Empty at the top
  // and at the bottom alike.
  ArrayRef<uint64_t> offsets() const { return Offsets; }

  // Join `Other` into this value. Return true if this value changed, which is
  // what keeps the worklist going.
  bool join(const TargetOffsets &Other) {
    if (Other.Top > Top) {
      Top = Other.Top;
      Offsets.clear();
      return true;
    }
    if (isTop())
      return false;

    bool Changed = Other.Incomplete && !Incomplete;
    Incomplete |= Other.Incomplete;
    for (uint64_t Offset : Other.Offsets) {
      if (is_contained(Offsets, Offset))
        continue;
      if (Offsets.size() == MaxSetPcTargets) {
        Top = Reason::TooMany;
        Offsets.clear();
        return true;
      }
      Offsets.push_back(Offset);
      Changed = true;
    }
    if (Changed)
      sort(Offsets);
    return Changed;
  }

  // Raise this value to unnameable. Return true if that changed it.
  bool raiseToUnnameable() { return join(unnameable()); }

private:
  // Why the value is at the top. A join keeps the larger reason, so a pair that
  // is both unnameable and over the cap reads as unnameable.
  enum class Reason { None, TooMany, Unnameable };

  SmallVector<uint64_t> Offsets;
  Reason Top = Reason::None;
  // Some path has not completed its split address calculation.
  bool Incomplete = false;
};

// What every register pair holds at one point, keyed by the pair's low register
// index. A pair with no entry is one the surrounding code says nothing about.
using PairOffsets = DenseMap<unsigned, TargetOffsets>;

// What the analysis knows about one scalar register within a block.
struct RegState {
  // Whether the block writes this register at all.
  bool Written = false;
  // The constant that the block folded into this register.
  std::optional<uint32_t> Constant;
  // The source offset carried by the pair that starts at this register.
  std::optional<uint64_t> PairOffset;
  enum AddressKind { Complete, PendingCarry, Incomplete };
  AddressKind PairAddressKind = Complete;
};

// What the analysis knows about the scalar registers while walking one block.
// Everything here is block-local: an offset is only followed from the capture
// that starts it to the transfer that reads it, both within one block. What a
// block leaves behind reaches the next one through the dataflow below, not
// through this state.
struct BlockState {
  // Record that the block writes the register at `Idx`. The write drops the
  // constant that the register held and the offsets of both pairs it belongs
  // to, the pair it starts and the pair it ends. The constant at `Idx - 1`
  // belongs to a different register and stands.
  void writeReg(unsigned Idx) {
    RegState &Reg = Regs[Idx];
    Reg.Written = true;
    Reg.Constant.reset();
    Reg.PairOffset.reset();
    Reg.PairAddressKind = RegState::Complete;
    if (Idx > 0)
      Regs[Idx - 1].PairOffset.reset();
    // A pending carry belongs to the value that the low add produced, and this
    // write replaces that value.
    if (CarryPair == Idx)
      CarryPair.reset();
  }

  // Whether the block writes either half of the pair that starts at `Idx`.
  bool pairIsWritten(unsigned Idx) const {
    return Regs.lookup(Idx).Written || Regs.lookup(Idx + 1).Written;
  }

  // Record what every pair the block writes holds where the block ends: the
  // source offset the pair carries, or an unnameable value when the writes left
  // no offset behind. Pairs the block leaves alone stay out of `Out`, because
  // they keep whatever reached the block.
  void summarize(PairOffsets &Out) const {
    auto Record = [&](unsigned Base) {
      std::optional<uint64_t> Offset = Regs.lookup(Base).PairOffset;
      Out[Base] = Offset ? TargetOffsets::single(
                               *Offset, Regs.lookup(Base).PairAddressKind !=
                                            RegState::Complete)
                         : TargetOffsets::unnameable();
    };
    for (const auto &Reg : Regs) {
      if (!Reg.second.Written)
        continue;
      Record(Reg.first);
      if (Reg.first > 0)
        Record(Reg.first - 1);
    }
  }

  // Everything the block did to each register, keyed by register index.
  // Registers the block has not touched read back as a default RegState.
  DenseMap<unsigned, RegState> Regs;
  // The pair whose low-half add left its carry in SCC, for as long as SCC
  // still holds it. Only the high-half add of that same pair may consume it.
  std::optional<unsigned> CarryPair;
};

// Record the write to every scalar register that `Di` defines. The walk below
// returns before reaching this for every instruction it models, so whatever
// gets here has an unknown effect on a captured program counter.
void writeDefs(const DecodedInst &Di, const MCRegisterInfo &MRI,
               BlockState &State) {
  for (unsigned I = 0; I != Di.NumDefs && I != Di.numOperands(); ++I) {
    if (!Di.isReg(I))
      continue;
    MCRegister Reg = Di.getReg(I);
    // A destination wider than one register covers several of the indices that
    // key the state, so writing it writes each register it spans.
    if (std::optional<unsigned> Idx = sgprIndex(MRI, Reg))
      State.writeReg(*Idx);
    for (MCPhysReg Sub : MRI.subregs(Reg))
      if (std::optional<unsigned> Idx = sgprIndex(MRI, Sub))
        State.writeReg(*Idx);
  }
}

// The value of operand `Index` when it is a compile-time constant. Every
// operand this is asked about belongs to a 32-bit scalar add, so a constant
// there always fits in 32 bits, signed or unsigned.
std::optional<uint32_t> immediate32(const MCInst &Inst, unsigned Index) {
  std::optional<int64_t> Value = evalOperandAsConst(Inst, Index);
  if (!Value)
    return std::nullopt;
  assert((isInt<32>(*Value) || isUInt<32>(*Value)) &&
         "32-bit scalar operand holds a value wider than 32 bits");
  return static_cast<uint32_t>(*Value);
}

// Whether `Op` transfers control through a register value.
bool isRegisterIndirectTransfer(CanonicalOp Op) {
  return Op == CanonicalOp::S_SETPC_B64 || Op == CanonicalOp::S_SWAPPC_B64;
}

// The destination register index of an instruction that writes one scalar
// destination, or nullopt when the destination is not a register the analysis
// follows.
std::optional<unsigned> destIndex(const DecodedInst &Di,
                                  const MCRegisterInfo &MRI) {
  assert(Di.NumDefs >= 1 && Di.numOperands() >= 1 && Di.isReg(0) &&
         "scalar destination expected in operand 0");
  return sgprIndex(MRI, Di.getReg(0));
}

// Every offset a transfer at a given source offset has been found to reach,
// keyed by the transfer's source offset.
using TransferTargets = DenseMap<uint64_t, DenseSet<uint64_t>>;

// Work out where every register-indirect transfer in `Insts` leads, splitting
// the walk at `WalkBlockStarts` and drawing the edges out of a transfer from
// `KnownTargets`. `InstOffsets` holds the offset of every decoded instruction.
// Control enters at `EntryOffset`, which must lead one of the blocks. `Insts`
// must not be empty.
Expected<SetPcAnalysis> classifyAgainstBlockStarts(
    ArrayRef<DecodedInst> Insts, const DenseSet<uint64_t> &WalkBlockStarts,
    const DenseSet<uint64_t> &InstOffsets, const TransferTargets &KnownTargets,
    uint64_t EntryOffset, const MCRegisterInfo &MRI) {
  assert(!Insts.empty() && "the walk needs at least one instruction to split");
  assert(Insts.size() <= std::numeric_limits<unsigned>::max() &&
         "instruction and block indices are tracked as unsigned");

  // One recovered block, in source order.
  struct Block {
    // Source offset of the first instruction of the block.
    uint64_t Start = 0;
    // Index into `Insts` of the last instruction of the block.
    unsigned LastInst = 0;
    // What the block itself leaves in the register pairs it writes. Pairs it
    // does not write are absent and keep what reached the block.
    PairOffsets Writes;
    // Source offsets of the blocks control can go to from here.
    SmallVector<uint64_t> Successors;
    // What the paths into the block leave in the register pairs, joined. Only
    // meaningful once `Reachable` is set.
    PairOffsets Entry;
    // Whether the dataflow has reached this block over any path.
    bool Reachable = false;

    // Join what a predecessor leaves into `Entry`. Return true if `Entry`
    // changed. A register pair only one side names is one the other side says
    // nothing about, which is as incomplete as a write that left no offset
    // behind.
    bool joinEntry(const PairOffsets &FromPredecessor) {
      if (!Reachable) {
        Reachable = true;
        Entry = FromPredecessor;
        return true;
      }

      bool Changed = false;
      for (auto &Held : Entry) {
        auto It = FromPredecessor.find(Held.first);
        Changed |= It == FromPredecessor.end() ? Held.second.raiseToUnnameable()
                                               : Held.second.join(It->second);
      }
      for (const auto &Held : FromPredecessor) {
        if (Entry.count(Held.first))
          continue;
        TargetOffsets Offsets = Held.second;
        Offsets.raiseToUnnameable();
        Entry[Held.first] = std::move(Offsets);
        Changed = true;
      }
      return Changed;
    }
  };
  SmallVector<Block> Blocks;
  // Index into `Blocks` of the block starting at a source offset.
  DenseMap<uint64_t, unsigned> BlockOf;

  // A transfer whose own block leaves the register pair it reads alone. It
  // waits for the dataflow to say what the paths into the block leave there.
  struct DeferredSite {
    uint64_t Offset;
    unsigned BlockIndex;
    unsigned PairBase;
  };
  SmallVector<DeferredSite> Deferred;

  SetPcAnalysis Result;

  // Resolve one transfer against what its own block built, deferring it when
  // the block says nothing about the pair it reads.
  auto resolveTransfer = [&](const DecodedInst &Di, const BlockState &State) {
    unsigned SourceIndex = Di.FirstSrcIdx;
    std::optional<unsigned> Source;
    if (SourceIndex < Di.numOperands() && Di.isReg(SourceIndex))
      Source = sgprIndex(MRI, Di.getReg(SourceIndex));
    if (!Source) {
      Result.Sites.insert(
          {Di.Offset, SetPcUnresolvable{SetPcRefusal::NotARegisterPair, 0}});
      return;
    }

    std::optional<uint64_t> Offset = State.Regs.lookup(*Source).PairOffset;
    if (!Offset) {
      if (!State.pairIsWritten(*Source)) {
        Deferred.push_back(
            {Di.Offset, static_cast<unsigned>(Blocks.size() - 1), *Source});
        return;
      }
      Result.Sites.insert(
          {Di.Offset,
           SetPcUnresolvable{SetPcRefusal::PairWrittenWithoutOffset, *Source}});
      return;
    }

    if (!InstOffsets.contains(*Offset)) {
      Result.Sites.insert(
          {Di.Offset,
           SetPcUnresolvable{SetPcRefusal::TargetNotAnInstruction, *Offset}});
      return;
    }

    Result.Sites.insert({Di.Offset, SetPcResolved{{*Offset}}});
    Result.ExtraBlockStarts.insert(*Offset);
  };

  BlockState State;
  for (unsigned I = 0, E = Insts.size(); I != E; ++I) {
    const DecodedInst &Di = Insts[I];
    if (Blocks.empty() || WalkBlockStarts.contains(Di.Offset)) {
      if (!Blocks.empty())
        State.summarize(Blocks.back().Writes);
      State = BlockState();
      BlockOf[Di.Offset] = Blocks.size();
      Blocks.push_back(Block());
      Blocks.back().Start = Di.Offset;
    }
    Blocks.back().LastInst = I;

    // SCC holds the carry out of a low-half add only until the next
    // instruction writes it, so anything that writes SCC ends the relation.
    // Only which pair owns the carry matters here; its value depends on the
    // program counter, which the analysis never knows. The low-half add below
    // re-establishes ownership for the pair it just wrote.
    std::optional<unsigned> Carry = State.CarryPair;
    if (Di.defsScc())
      State.CarryPair.reset();

    switch (Di.CanonOp) {
    case CanonicalOp::S_GETPC_B64: {
      // The capture names the offset of the instruction that follows it.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      State.writeReg(*Dst);
      State.writeReg(*Dst + 1);
      State.Regs[*Dst].PairOffset = Di.Offset + Di.sizeInBytes();
      continue;
    }

    case CanonicalOp::S_ADD_U32: {
      // A displacement added to the offset that a pair already carries, or a
      // plain constant fold that a later add can use as its addend.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      assert(Src1 < Di.numOperands() && "scalar add has two source operands");
      std::optional<uint32_t> Src0Imm = immediate32(Di.Inst, Src0);
      std::optional<uint32_t> Src1Imm = immediate32(Di.Inst, Src1);
      if (Src0Imm && Src1Imm) {
        // A fold of two constants. No displacement chain can continue from its
        // carry out, and the clear above already gave up ownership of SCC, so
        // only the folded value is worth recording.
        State.writeReg(*Dst);
        State.Regs[*Dst].Constant = *Src0Imm + *Src1Imm;
        continue;
      }
      std::optional<unsigned> Src0Idx;
      if (Di.isReg(Src0))
        Src0Idx = sgprIndex(MRI, Di.getReg(Src0));
      if (Src0Idx != Dst)
        break;
      std::optional<uint64_t> Base = State.Regs.lookup(*Dst).PairOffset;
      if (!Base)
        break;
      std::optional<uint32_t> Addend = Src1Imm;
      if (!Addend && Di.isReg(Src1)) {
        std::optional<unsigned> Src1Idx = sgprIndex(MRI, Di.getReg(Src1));
        if (Src1Idx)
          Addend = State.Regs.lookup(*Src1Idx).Constant;
      }
      if (!Addend)
        break;
      uint64_t Target = *Base + *Addend;
      RegState::AddressKind Kind = State.Regs.lookup(*Dst).PairAddressKind;
      State.writeReg(*Dst);
      State.Regs[*Dst].PairOffset = Target;
      State.Regs[*Dst].PairAddressKind = Kind == RegState::Complete
                                             ? RegState::PendingCarry
                                             : RegState::Incomplete;
      // A high-half add may now consume the carry out of this add.
      State.CarryPair = *Dst;
      continue;
    }

    case CanonicalOp::S_ADDC_U32: {
      // High half of a split displacement. This adds SCC, so it continues the
      // low half only while SCC still holds that half's carry, and `Carry`
      // names the pair that carry belongs to. Everything else that writes the
      // same high register is an ordinary write, whichever pair it sits in.
      //
      // The low add already folded the whole displacement into the offset
      // whenever the source offset stays within four gigabytes of the capture,
      // so this only adds the high addend.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Carry || Dst != *Carry + 1)
        break;
      unsigned Low = *Carry;
      std::optional<uint64_t> Target = State.Regs.lookup(Low).PairOffset;
      if (!Target)
        break;
      unsigned Src0 = Di.FirstSrcIdx;
      unsigned Src1 = Src0 + 1;
      assert(Src1 < Di.numOperands() && "scalar add has two source operands");
      if (!Di.isReg(Src0) || sgprIndex(MRI, Di.getReg(Src0)) != Dst)
        break;
      std::optional<uint32_t> Addend = immediate32(Di.Inst, Src1);
      if (!Addend)
        break;
      RegState::AddressKind Kind = State.Regs.lookup(Low).PairAddressKind;
      State.writeReg(*Dst);
      State.Regs[Low].PairOffset =
          *Target + (static_cast<uint64_t>(*Addend) << 32);
      State.Regs[Low].PairAddressKind = Kind == RegState::PendingCarry
                                            ? RegState::Complete
                                            : RegState::Incomplete;
      continue;
    }

    case CanonicalOp::S_ADD_NC_U64: {
      // The whole displacement folded into one add. It commutes, so the pair
      // carrying the offset stands on either side of the constant.
      std::optional<unsigned> Dst = destIndex(Di, MRI);
      if (!Dst)
        break;
      std::optional<uint64_t> Base = State.Regs.lookup(*Dst).PairOffset;
      if (!Base || Di.SrcMap.size() < 2)
        break;
      unsigned SrcA = Di.SrcMap[0];
      unsigned SrcB = Di.SrcMap[1];
      std::optional<unsigned> SrcAIdx;
      if (Di.isReg(SrcA))
        SrcAIdx = sgprIndex(MRI, Di.getReg(SrcA));
      std::optional<unsigned> SrcBIdx;
      if (Di.isReg(SrcB))
        SrcBIdx = sgprIndex(MRI, Di.getReg(SrcB));
      std::optional<int64_t> Displacement;
      if (SrcAIdx == Dst && !SrcBIdx)
        Displacement = evalOperandAsConst(Di.Inst, SrcB);
      else if (SrcBIdx == Dst && !SrcAIdx)
        Displacement = evalOperandAsConst(Di.Inst, SrcA);
      if (!Displacement)
        break;
      uint64_t Target = *Base + static_cast<uint64_t>(*Displacement);
      RegState::AddressKind Kind = State.Regs.lookup(*Dst).PairAddressKind;
      State.writeReg(*Dst);
      State.writeReg(*Dst + 1);
      State.Regs[*Dst].PairOffset = Target;
      State.Regs[*Dst].PairAddressKind = Kind;
      continue;
    }

    case CanonicalOp::S_SETPC_B64:
      resolveTransfer(Di, State);
      continue;

    case CanonicalOp::S_SWAPPC_B64: {
      resolveTransfer(Di, State);
      // The call writes the offset it returns to over whatever its destination
      // pair held, which is a source offset like any a displacement computes.
      if (std::optional<unsigned> Dst = destIndex(Di, MRI)) {
        State.writeReg(*Dst);
        State.writeReg(*Dst + 1);
        uint64_t Return = Di.Offset + Di.sizeInBytes();
        if (InstOffsets.contains(Return))
          State.Regs[*Dst].PairOffset = Return;
      }
      continue;
    }

    default:
      break;
    }

    writeDefs(Di, MRI, State);
  }
  State.summarize(Blocks.back().Writes);

  // Draw the edges out of every block. A register-indirect transfer names no
  // offset the decode could follow, so its edges are the targets already found
  // for it: what this walk resolved, plus what the earlier walks reached.
  //
  // A transfer nothing has resolved yet draws no edges at all, which is
  // optimistic: the blocks it reaches miss what it would have left them. That
  // is sound only because a transfer still unresolved once the rounds settle
  // refuses the whole kernel, so nothing computed from the missing edges
  // reaches the raise.
  for (unsigned I = 0, E = Blocks.size(); I != E; ++I) {
    Block &Blk = Blocks[I];
    const DecodedInst &Last = Insts[Blk.LastInst];
    if (isRegisterIndirectTransfer(Last.CanonOp)) {
      auto Site = Result.Sites.find(Last.Offset);
      if (Site != Result.Sites.end())
        if (const auto *Resolved = std::get_if<SetPcResolved>(&Site->second))
          Blk.Successors.assign(Resolved->Targets);
      auto Known = KnownTargets.find(Last.Offset);
      if (Known != KnownTargets.end())
        for (uint64_t Target : Known->second)
          if (!is_contained(Blk.Successors, Target))
            Blk.Successors.push_back(Target);
      continue;
    }
    std::optional<uint64_t> Next;
    if (I + 1 != E)
      Next = Blocks[I + 1].Start;
    Expected<SmallVector<uint64_t>> Successors =
        computeDecodedBlockSuccessors(Last, Next);
    if (!Successors)
      return Successors.takeError();
    Blk.Successors = std::move(*Successors);
  }

  // Run the forward dataflow to a fixpoint. The lattice is bounded: a register
  // pair holds at most MaxSetPcTargets offsets before it goes to the top, and a
  // join only ever moves a value up, so the worklist runs dry.
  //
  // Control enters at the entry block, which need not be the lowest-addressed
  // one: a callee followed into the decode may sit below its caller.
  assert(BlockOf.contains(EntryOffset) && "the entry offset leads a block");
  unsigned EntryBlock = BlockOf.at(EntryOffset);
  Blocks[EntryBlock].Reachable = true;
  SetVector<unsigned> Worklist;
  Worklist.insert(EntryBlock);
  while (!Worklist.empty()) {
    unsigned I = Worklist.pop_back_val();

    PairOffsets Exit = Blocks[I].Entry;
    for (const auto &Written : Blocks[I].Writes)
      Exit[Written.first] = Written.second;

    for (uint64_t Successor : Blocks[I].Successors) {
      auto It = BlockOf.find(Successor);
      if (It == BlockOf.end())
        continue;
      if (Blocks[It->second].joinEntry(Exit))
        Worklist.insert(It->second);
    }
  }

  for (const Block &Blk : Blocks)
    if (Blk.Reachable)
      for (const auto &[Base, Offsets] : Blk.Entry)
        if (Offsets.offsets().size() == 1 && !Offsets.isIncomplete())
          Result.EntryOffsets[Blk.Start][Base] = Offsets.offsets().front();

  // A deferred transfer reads a register pair its own block leaves alone, so
  // what the paths into the block leave there is what the transfer reads.
  for (const DeferredSite &Site : Deferred) {
    const Block &Blk = Blocks[Site.BlockIndex];
    auto Refuse = [&](SetPcRefusal Why, uint64_t Subject) {
      Result.Sites[Site.Offset] = SetPcUnresolvable{Why, Subject};
    };

    if (!Blk.Reachable) {
      Refuse(SetPcRefusal::BlockUnreachable, Site.PairBase);
      continue;
    }

    auto Held = Blk.Entry.find(Site.PairBase);
    if (Held == Blk.Entry.end()) {
      Refuse(SetPcRefusal::PairNeverHeldOffset, Site.PairBase);
      continue;
    }

    const TargetOffsets &Offsets = Held->second;
    if (Offsets.isUnnameable()) {
      Refuse(SetPcRefusal::PairWrittenWithoutOffset, Site.PairBase);
      continue;
    }
    if (Offsets.isTooMany()) {
      Refuse(SetPcRefusal::TooManyTargets, Site.PairBase);
      continue;
    }

    assert(!Offsets.offsets().empty() &&
           "a reachable block joins at least one predecessor, which either "
           "names an offset or raises the pair to the top");
    const uint64_t *Undecoded = find_if(Offsets.offsets(), [&](uint64_t Value) {
      return !InstOffsets.contains(Value);
    });
    if (Undecoded != Offsets.offsets().end()) {
      Refuse(SetPcRefusal::TargetNotAnInstruction, *Undecoded);
      continue;
    }

    SetPcResolved Resolved;
    Resolved.Targets.assign(Offsets.offsets());
    Result.Sites[Site.Offset] = std::move(Resolved);
    Result.ExtraBlockStarts.insert_range(Offsets.offsets());
  }

  return Result;
}

} // namespace

Expected<SetPcAnalysis> analyzeSetPc(ArrayRef<DecodedInst> Insts,
                                     const std::set<uint64_t> &BlockStarts,
                                     uint64_t EntryOffset, const MCState &Mc) {
  SetPcAnalysis Result;
  if (Insts.empty())
    return Result;

  DenseSet<uint64_t> InstOffsets;
  InstOffsets.reserve(Insts.size());
  for (const DecodedInst &Di : Insts)
    InstOffsets.insert(Di.Offset);

  // A transfer ends the block it sits in, so whatever follows it leads a block
  // of its own. For a call that block is where the callee returns to; for a
  // jump nothing falls into it, but the instructions there still need a block
  // to be raised into. The offset one past the last instruction leads nothing.
  //
  // Splitting here also keeps each transfer at the end of its block, so a
  // transfer reads exactly the state that the instructions before it left
  // behind.
  DenseSet<uint64_t> WalkBlockStarts(llvm::from_range, BlockStarts);
  // Control enters here, so this leads a block however the decode was split.
  WalkBlockStarts.insert(EntryOffset);
  for (const DecodedInst &Di : Insts) {
    if (!isRegisterIndirectTransfer(Di.CanonOp))
      continue;
    uint64_t Fallthrough = Di.Offset + Di.sizeInBytes();
    if (InstOffsets.contains(Fallthrough))
      WalkBlockStarts.insert(Fallthrough);
  }

  // Resolving a transfer both names a block the walk did not split at and draws
  // an edge the walk did not have. Either changes what reaches the other
  // transfers, so the walk repeats until it learns nothing new. Both the block
  // starts and the edges only ever grow, so a round that learns nothing is the
  // fixpoint.
  //
  // `KnownTargets` keeps every edge any round drew, including edges out of a
  // transfer a later round stops resolving. A stale edge can carry offsets into
  // a block along a path the final classification no longer believes in. That
  // is harmless only because the transfer that stopped resolving refuses the
  // whole kernel, so no answer computed from its stale edge reaches the raise.
  //
  // That growth also bounds the number of rounds, and a walk that runs past the
  // bound has broken the argument. Refusing the kernel beats spinning on it.
  const size_t MaxRounds = Insts.size() * (MaxSetPcTargets + 1) + 1;
  TransferTargets KnownTargets;
  for (size_t Round = 0;; ++Round) {
    if (Round == MaxRounds)
      return createStringError(
          inconvertibleErrorCode(),
          "program-counter analysis did not settle in %zu rounds", MaxRounds);

    Expected<SetPcAnalysis> Classified =
        classifyAgainstBlockStarts(Insts, WalkBlockStarts, InstOffsets,
                                   KnownTargets, EntryOffset, *Mc.RegInfo);
    if (!Classified)
      return Classified.takeError();

    bool Learned = false;
    for (uint64_t Start : Classified->ExtraBlockStarts)
      Learned |= WalkBlockStarts.insert(Start).second;
    for (const auto &Site : Classified->Sites)
      if (const auto *Resolved = std::get_if<SetPcResolved>(&Site.second))
        for (uint64_t Target : Resolved->Targets)
          Learned |= KnownTargets[Site.first].insert(Target).second;
    if (Learned)
      continue;

    // The raise must build the blocks the analysis walked, or its answers stop
    // holding. Report every split the walk made that the decode did not.
    Result = std::move(*Classified);
    Result.ExtraBlockStarts = std::move(WalkBlockStarts);
    for (uint64_t Start : BlockStarts)
      Result.ExtraBlockStarts.erase(Start);
    return Result;
  }
}

} // namespace COMGR::transpiler
