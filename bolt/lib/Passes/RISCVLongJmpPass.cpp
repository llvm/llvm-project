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

using namespace llvm;
using namespace llvm::bolt;

namespace {

/// Return scratch candidates among caller-saved GPRs, including argument
/// registers, excluding special registers and registers referenced by CFI.
BitVector getScratchRegCandidates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  BitVector Regs(BC.MRI->getNumRegs());
  BC.MIB->getGPRegs(Regs, /*IncludeAlias=*/false);
  BC.MIB->removeNonScavengeableRegs(Regs);
  BitVector CalleeSaved(Regs.size());
  BC.MIB->getCalleeSavedRegs(CalleeSaved);
  Regs.reset(CalleeSaved);

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

/// Insert a stub for every cross-fragment CFG edge, without estimating its
/// final distance. Return false if splitting was cancelled for lack of scratch.
/// Analyze all edges before mutating the CFG so fallback needs no stub
/// rollback.
bool relaxFunction(BinaryFunction &BF, RegAnalysis &RA) {
  BinaryContext &BC = BF.getBinaryContext();
  BitVector Candidates = getScratchRegCandidates(BF);
  struct LongJumpEdge {
    BinaryBasicBlock *Source;
    BinaryBasicBlock *Target;
    BinaryBasicBlock *Stub;
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
          return false;
        }
        ScratchRegs[Target] = Dead.find_first();
      }
    }
  }

  // splitEdge() assumes at most two fragments, so insert the stubs here.
  for (LongJumpEdge &E : Edges) {
    auto Stub = BF.createBasicBlock();
    E.Stub = Stub.get();
    const auto BI = E.Source->getBranchInfo(*E.Target);
    Stub->setFragmentNum(E.Source->getFragmentNum());
    Stub->setExecutionCount(BI.Count);
    Stub->addSuccessor(E.Target, BI.Count, BI.MispredictedCount);
    E.Source->replaceSuccessor(E.Target, Stub.get(), BI.Count,
                               BI.MispredictedCount);
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBlocks;
    NewBlocks.push_back(std::move(Stub));
    BF.insertBasicBlocks(E.Source, std::move(NewBlocks),
                         /*UpdateLayout=*/true, /*UpdateCFIState=*/true,
                         /*RecomputeLandingPads=*/false);
  }
  BF.fixBranches();
  for (const LongJumpEdge &E : Edges) {
    MCInst Jump;
    // PseudoJump expands to AUIPC scratch + JALR x0, scratch. It overwrites the
    // chosen dead register without changing ra; JITLink may later shorten it.
    BC.MIB->createLongBranch(Jump, E.Target->getLabel(), ScratchRegs[E.Target],
                             BC.Ctx.get());
    E.Stub->clear();
    E.Stub->addInstruction(Jump);
  }
  return true;
}

} // namespace

Error RISCVLongJmpPass::runOnFunctions(BinaryContext &BC) {
  RegAnalysis RA(BC, nullptr, nullptr);
  unsigned Relaxed = 0, KeptTogether = 0;
  for (auto &[Address, BF] : BC.getBinaryFunctions()) {
    if (!BC.shouldEmit(BF) || !BF.isSimple() || !BF.isSplit())
      continue;
    if (relaxFunction(BF, RA))
      ++Relaxed;
    else
      ++KeptTogether;
  }
  BC.outs() << "BOLT-INFO: RISC-V relaxed branches in " << Relaxed
            << " split functions; kept " << KeptTogether
            << " functions together without a dead scratch register\n";
  return Error::success();
}
