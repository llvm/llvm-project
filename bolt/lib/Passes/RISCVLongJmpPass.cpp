//===- RISCVLongJmpPass.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVLongJmpPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RISCVLongJmpPass.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/RegAnalysis.h"
#include <optional>

using namespace llvm;
using namespace llvm::bolt;

namespace {

/// Return scratch candidates among caller-saved GPRs, including argument
/// registers, excluding special registers and registers referenced by CFI.
BitVector getScratchRegCandidates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  BitVector Regs(BC.MRI->getNumRegs());
  BC.MIB->getLongJumpScratchRegs(Regs);

  // Ordinary liveness does not account for registers needed by stack unwinding.
  // Conservatively exclude CFI references throughout the function instead of
  // interpreting the unwind state separately at every prospective stub.
  auto removeCFIRegs = [&](const MCCFIInstruction &CFI) {
    switch (CFI.getOperation()) {
    case MCCFIInstruction::OpEscape:
      // Raw DWARF may refer to registers that this scan cannot identify.
      return false;
    case MCCFIInstruction::OpRegister:
      if (auto Reg = BC.MRI->getLLVMRegNum(CFI.getRegister2(), false))
        Regs.reset(*Reg);
      [[fallthrough]];
    case MCCFIInstruction::OpDefCfa:
    case MCCFIInstruction::OpDefCfaRegister:
    case MCCFIInstruction::OpLLVMDefAspaceCfa:
    case MCCFIInstruction::OpOffset:
    case MCCFIInstruction::OpRelOffset:
    case MCCFIInstruction::OpValOffset:
    case MCCFIInstruction::OpRestore:
    case MCCFIInstruction::OpUndefined:
    case MCCFIInstruction::OpSameValue:
      if (auto Reg = BC.MRI->getLLVMRegNum(CFI.getRegister(), false))
        Regs.reset(*Reg);
      break;
    default:
      break;
    }
    return true;
  };
  for (const MCCFIInstruction &CFI : BF.cie())
    if (!removeCFIRegs(CFI))
      return BitVector(Regs.size());
  for (const BinaryBasicBlock &BB : BF)
    for (const MCInst &Inst : BB)
      if (const MCCFIInstruction *CFI = BF.getCFIFor(Inst))
        if (!removeCFIRegs(*CFI))
          return BitVector(Regs.size());
  return Regs;
}

/// Cancel splitting while preserving the current basic block layout order.
/// This avoids introducing a long jump that clobbers a required register.
void keepFragmentsTogether(BinaryFunction &BF) {
  BinaryFunction::BasicBlockOrderType Order(BF.getLayout().block_begin(),
                                            BF.getLayout().block_end());
  for (BinaryBasicBlock *BB : Order)
    BB->setFragmentNum(FragmentNum::main());
  BF.getLayout().update(Order);
  if (BF.hasEHRanges())
    BF.setLPFragment(FragmentNum::main(), FragmentNum::main());
  BF.fixBranches();
}

/// Bound distances from each block to the end of its fragment, including all
/// long-jump stubs. Allow for alignment, an 8-byte relaxed conditional branch,
/// and a 4-byte jump from fixBranches() per block.
DenseMap<const BinaryBasicBlock *, uint64_t>
estimateDistancesToFragmentEnd(const BinaryFunction &BF) {
  DenseMap<const BinaryBasicBlock *, uint64_t> DistanceToEnd;
  for (const FunctionFragment &FF : BF.getLayout().fragments()) {
    uint64_t Size = 0;
    for (const BinaryBasicBlock *BB : FF)
      for (const BinaryBasicBlock *Target : BB->successors())
        if (Target->getFragmentNum() != FF.getFragmentNum())
          Size += 8;
    for (const BinaryBasicBlock *BB : llvm::reverse(FF)) {
      Size += BB->estimateSize() + 12 + BB->getAlignment() - 1;
      DistanceToEnd[BB] = Size;
    }
  }
  return DistanceToEnd;
}

/// Insert a stub for every cross-fragment CFG edge. Return the number of stubs,
/// or std::nullopt if splitting was cancelled for lack of scratch.
/// Analyze all edges before mutating the CFG so fallback needs no stub
/// rollback.
std::optional<size_t> insertLongJumpStubs(BinaryFunction &BF, RegAnalysis &RA) {
  BinaryContext &BC = BF.getBinaryContext();
  BitVector Candidates = getScratchRegCandidates(BF);
  struct LongJumpEdge {
    BinaryBasicBlock *Source;
    BinaryBasicBlock *Target;
    BinaryBasicBlock *Stub;
    bool AtFragmentEnd = false;
  };
  SmallVector<LongJumpEdge> Edges;
  DenseMap<BinaryBasicBlock *, MCPhysReg> ScratchRegs;
  {
    DataflowInfoManager DIM(BF, &RA, nullptr);
    LivenessAnalysis &LA = DIM.getLivenessAnalysis();
    for (BinaryBasicBlock &BB : BF) {
      for (BinaryBasicBlock *Target : BB.successors()) {
        if (Target->getFragmentNum() == BB.getFragmentNum())
          continue;
        Edges.push_back({&BB, Target, nullptr});
        if (ScratchRegs.contains(Target))
          continue;
        BitVector Dead = *LA.getStateAt(ProgramPoint::getFirstPointAt(*Target));
        Dead.flip();
        Dead &= Candidates;
        // There is no ABI-reserved temporary for arbitrary intra-function
        // branches on RISC-V. Never insert a clobber when no register is dead.
        if (Dead.none()) {
          BC.errs() << "BOLT-WARNING: keeping " << BF
                    << " unsplit: no dead register for a RISC-V long jump\n";
          DIM.invalidateLivenessAnalysis();
          keepFragmentsTogether(BF);
          return std::nullopt;
        }
        ScratchRegs[Target] = Dead.find_first();
      }
    }
  }

  const DenseMap<const BinaryBasicBlock *, uint64_t> DistanceToEnd =
      estimateDistancesToFragmentEnd(BF);
  // Keep cold stubs at the end of the fragment to preserve hot fall-through
  // paths, provided they fit in the 4 KiB conditional-branch range.
  // Choose positions before inserting stubs changes the layout.
  for (LongJumpEdge &E : Edges) {
    const BinaryBasicBlock *Next =
        BF.getLayout().getBasicBlockAfter(E.Source, /*IgnoreSplits=*/false);
    E.AtFragmentEnd = E.Source->getBranchInfo(*E.Target).Count == 0 &&
                      E.Source->succ_size() == 2 &&
                      E.Source->isSuccessor(Next) &&
                      DistanceToEnd.lookup(E.Source) <= 4094;
  }

  // splitEdge() assumes at most two fragments, so insert the stubs here.
  for (LongJumpEdge &E : Edges) {
    auto Stub = BF.createBasicBlock();
    E.Stub = Stub.get();
    const auto BI = E.Source->getBranchInfo(*E.Target);
    Stub->setFragmentNum(E.Source->getFragmentNum());
    Stub->setCFIState(E.Source->getCFIStateAtExit());
    Stub->setExecutionCount(BI.Count);
    Stub->addSuccessor(E.Target, BI.Count, BI.MispredictedCount);
    BinaryBasicBlock *InsertAfter = E.Source;
    if (E.AtFragmentEnd)
      InsertAfter =
          BF.getLayout().getFragment(E.Source->getFragmentNum()).back();
    E.Source->replaceSuccessor(E.Target, Stub.get(), BI.Count,
                               BI.MispredictedCount);
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBlocks;
    NewBlocks.push_back(std::move(Stub));
    BF.insertBasicBlocks(InsertAfter, std::move(NewBlocks),
                         /*UpdateLayout=*/true, /*UpdateCFIState=*/false,
                         /*RecomputeLandingPads=*/false);
  }
  BF.fixBranches();
  for (const LongJumpEdge &E : Edges) {
    MCInst Jump;
    // PseudoJump expands to AUIPC scratch + JALR x0, scratch. It overwrites the
    // chosen dead register without changing ra; JITLink may later shorten it.
    BC.MIB->createLongUncondBranch(Jump, E.Target->getLabel(),
                                   ScratchRegs[E.Target], BC.Ctx.get());
    E.Stub->clear();
    E.Stub->addInstruction(Jump);
  }
  return Edges.size();
}

} // namespace

Error RISCVLongJmpPass::runOnFunctions(BinaryContext &BC) {
  RegAnalysis RA(BC, nullptr, nullptr);
  unsigned FunctionsWithStubs = 0, KeptTogether = 0;
  for (auto &[Address, BF] : BC.getBinaryFunctions()) {
    if (!BC.shouldEmit(BF) || !BF.isSimple() || !BF.isSplit())
      continue;
    if (std::optional<size_t> NumStubs = insertLongJumpStubs(BF, RA)) {
      if (*NumStubs)
        ++FunctionsWithStubs;
    } else {
      ++KeptTogether;
    }
  }
  if (FunctionsWithStubs || KeptTogether)
    BC.outs() << "BOLT-INFO: RISC-V inserted long-jump stubs in "
              << FunctionsWithStubs << " split functions; kept " << KeptTogether
              << " functions together without a dead scratch register\n";
  return Error::success();
}
