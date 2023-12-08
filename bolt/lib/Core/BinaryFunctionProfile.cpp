//===- bolt/Core/BinaryFunctionProfile.cpp - Profile processing -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements BinaryFunction member functions related to processing
// the execution profile.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<IndirectCallPromotionType> ICP(
    "indirect-call-promotion", cl::init(ICP_NONE),
    cl::desc("indirect call promotion"),
    cl::values(
        clEnumValN(ICP_NONE, "none", "do not perform indirect call promotion"),
        clEnumValN(ICP_CALLS, "calls", "perform ICP on indirect calls"),
        clEnumValN(ICP_JUMP_TABLES, "jump-tables",
                   "perform ICP on jump tables"),
        clEnumValN(ICP_ALL, "all", "perform ICP on calls and jump tables")),
    cl::ZeroOrMore, cl::cat(BoltOptCategory));

static cl::alias ICPAlias("icp",
                          cl::desc("Alias for --indirect-call-promotion"),
                          cl::aliasopt(ICP));

extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<bool> FixFuncCounts(
    "fix-func-counts",
    cl::desc("adjust function counts based on basic blocks execution count"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> FixBlockCounts(
    "fix-block-counts",
    cl::desc("adjust block counts based on outgoing branch counts"),
    cl::init(true), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    InferFallThroughs("infer-fall-throughs",
                      cl::desc("infer execution count for fall-through blocks"),
                      cl::Hidden, cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

void BinaryFunction::postProcessProfile() {
  if (!hasValidProfile()) {
    clearProfile();
    return;
  }

  if (!(getProfileFlags() & PF_LBR))
    return;

  // If we have at least some branch data for the function indicate that it
  // was executed.
  if (opts::FixFuncCounts && ExecutionCount == 0)
    ExecutionCount = 1;

  // Compute preliminary execution count for each basic block.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    if ((!BB->isEntryPoint() && !BB->isLandingPad()) ||
        BB->ExecutionCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB->ExecutionCount = 0;
  }
  for (BinaryBasicBlock *BB : BasicBlocks) {
    auto SuccBIIter = BB->branch_info_begin();
    for (BinaryBasicBlock *Succ : BB->successors()) {
      // All incoming edges to the primary entry have been accounted for, thus
      // we skip the update here.
      if (SuccBIIter->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
          Succ != BasicBlocks.front())
        Succ->setExecutionCount(Succ->getExecutionCount() + SuccBIIter->Count);
      ++SuccBIIter;
    }
  }

  // Fix for old profiles.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    if (BB->size() != 1 || BB->succ_size() != 1)
      continue;

    if (BB->getKnownExecutionCount() == 0)
      continue;

    MCInst *Instr = BB->getFirstNonPseudoInstr();
    assert(Instr && "expected non-pseudo instr");
    if (!BC.MIB->hasAnnotation(*Instr, "NOP"))
      continue;

    BinaryBasicBlock *FTSuccessor = BB->getSuccessor();
    BinaryBasicBlock::BinaryBranchInfo &BI = BB->getBranchInfo(*FTSuccessor);
    if (!BI.Count) {
      BI.Count = BB->getKnownExecutionCount();
      FTSuccessor->setExecutionCount(FTSuccessor->getKnownExecutionCount() +
                                     BI.Count);
    }
  }

  if (opts::FixBlockCounts) {
    for (BinaryBasicBlock *BB : BasicBlocks) {
      // Make sure that execution count of a block is at least the branch count
      // of an incoming/outgoing jump.
      auto SuccBIIter = BB->branch_info_begin();
      for (BinaryBasicBlock *Succ : BB->successors()) {
        uint64_t Count = SuccBIIter->Count;
        if (Count != BinaryBasicBlock::COUNT_NO_PROFILE && Count > 0) {
          Succ->setExecutionCount(std::max(Succ->getExecutionCount(), Count));
          BB->setExecutionCount(std::max(BB->getExecutionCount(), Count));
        }
        ++SuccBIIter;
      }
      // Make sure that execution count of a block is at least the number of
      // function calls from the block.
      for (MCInst &Inst : *BB) {
        // Ignore non-call instruction
        if (!BC.MIB->isCall(Inst))
          continue;

        auto CountAnnt = BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "Count");
        if (CountAnnt)
          BB->setExecutionCount(std::max(BB->getExecutionCount(), *CountAnnt));
      }
    }
  }

  if (opts::InferFallThroughs)
    inferFallThroughCounts();

  // Update profile information for jump tables based on CFG branch data.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    const MCInst *LastInstr = BB->getLastNonPseudoInstr();
    if (!LastInstr)
      continue;
    const uint64_t JTAddress = BC.MIB->getJumpTable(*LastInstr);
    if (!JTAddress)
      continue;
    JumpTable *JT = getJumpTableContainingAddress(JTAddress);
    if (!JT)
      continue;

    uint64_t TotalBranchCount = 0;
    for (const BinaryBasicBlock::BinaryBranchInfo &BranchInfo :
         BB->branch_info()) {
      TotalBranchCount += BranchInfo.Count;
    }
    JT->Count += TotalBranchCount;

    if (opts::ICP < ICP_JUMP_TABLES && opts::JumpTables < JTS_AGGRESSIVE)
      continue;

    if (JT->Counts.empty())
      JT->Counts.resize(JT->Entries.size());
    auto EI = JT->Entries.begin();
    uint64_t Delta = (JTAddress - JT->getAddress()) / JT->EntrySize;
    EI += Delta;
    while (EI != JT->Entries.end()) {
      const BinaryBasicBlock *TargetBB = getBasicBlockForLabel(*EI);
      if (TargetBB) {
        const BinaryBasicBlock::BinaryBranchInfo &BranchInfo =
            BB->getBranchInfo(*TargetBB);
        assert(Delta < JT->Counts.size());
        JT->Counts[Delta].Count += BranchInfo.Count;
        JT->Counts[Delta].Mispreds += BranchInfo.MispredictedCount;
      }
      ++Delta;
      ++EI;
      // A label marks the start of another jump table.
      if (JT->Labels.count(Delta * JT->EntrySize))
        break;
    }
  }
}

void BinaryFunction::mergeProfileDataInto(BinaryFunction &BF) const {
  // No reason to merge invalid or empty profiles into BF.
  if (!hasValidProfile())
    return;

  // Update function execution count.
  if (getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE)
    BF.setExecutionCount(BF.getKnownExecutionCount() + getExecutionCount());

  // Since we are merging a valid profile, the new profile should be valid too.
  // It has either already been valid, or it has been cleaned up.
  BF.ProfileMatchRatio = 1.0f;

  // Update basic block and edge counts.
  auto BBMergeI = BF.begin();
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BinaryBasicBlock *BBMerge = &*BBMergeI;
    assert(getIndex(BB) == BF.getIndex(BBMerge));

    // Update basic block count.
    if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE) {
      BBMerge->setExecutionCount(BBMerge->getKnownExecutionCount() +
                                 BB->getExecutionCount());
    }

    // Update edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BIMergeI = BBMerge->branch_info_begin();
    auto BII = BB->branch_info_begin();
    for (const BinaryBasicBlock *BBSucc : BB->successors()) {
      (void)BBSucc;
      assert(getIndex(BBSucc) == BF.getIndex(*BBMergeSI));

      // At this point no branch count should be set to COUNT_NO_PROFILE.
      assert(BII->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");
      assert(BIMergeI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");

      BIMergeI->Count += BII->Count;

      // When we merge inferred and real fall-through branch data, the merged
      // data is considered inferred.
      if (BII->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED &&
          BIMergeI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        BIMergeI->MispredictedCount += BII->MispredictedCount;
      } else {
        BIMergeI->MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
      }

      ++BBMergeSI;
      ++BII;
      ++BIMergeI;
    }
    assert(BBMergeSI == BBMerge->succ_end());

    ++BBMergeI;
  }
  assert(BBMergeI == BF.end());

  // Merge jump tables profile info.
  auto JTMergeI = BF.JumpTables.begin();
  for (const auto &JTEntry : JumpTables) {
    if (JTMergeI->second->Counts.empty())
      JTMergeI->second->Counts.resize(JTEntry.second->Counts.size());
    auto CountMergeI = JTMergeI->second->Counts.begin();
    for (const JumpTable::JumpInfo &JI : JTEntry.second->Counts) {
      CountMergeI->Count += JI.Count;
      CountMergeI->Mispreds += JI.Mispreds;
      ++CountMergeI;
    }
    assert(CountMergeI == JTMergeI->second->Counts.end());

    ++JTMergeI;
  }
  assert(JTMergeI == BF.JumpTables.end());
}

void BinaryFunction::inferFallThroughCounts() {
  // Work on a basic block at a time, propagating frequency information
  // forwards.
  // It is important to walk in the layout order.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    const uint64_t BBExecCount = BB->getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (BB->succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data.
    uint64_t ReportedBranches = 0;
    for (const BinaryBasicBlock::BinaryBranchInfo &SuccBI : BB->branch_info())
      if (SuccBI.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        ReportedBranches += SuccBI.Count;

    // Get taken count of conditional tail call if the block ends with one.
    uint64_t CTCTakenCount = 0;
    const MCInst *CTCInstr = BB->getLastNonPseudoInstr();
    if (CTCInstr && BC.MIB->getConditionalTailCall(*CTCInstr)) {
      CTCTakenCount = BC.MIB->getAnnotationWithDefault<uint64_t>(
          *CTCInstr, "CTCTakenCount");
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (const BinaryBasicBlock *LP : BB->landing_pads())
      ReportedThrows += LP->getExecutionCount();

    const uint64_t TotalReportedJumps =
        ReportedBranches + CTCTakenCount + ReportedThrows;

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch.
    uint64_t Inferred = 0;
    if (BBExecCount > TotalReportedJumps)
      Inferred = BBExecCount - TotalReportedJumps;

    LLVM_DEBUG(
        if (BBExecCount < TotalReportedJumps) dbgs()
            << "Fall-through inference is slightly inconsistent. "
               "exec frequency is less than the outgoing edges frequency ("
            << BBExecCount << " < " << ReportedBranches
            << ") for  BB at offset 0x"
            << Twine::utohexstr(getAddress() + BB->getOffset()) << '\n';);

    if (BB->succ_size() <= 2) {
      // Skip if the last instruction is an unconditional jump.
      const MCInst *LastInstr = BB->getLastNonPseudoInstr();
      if (LastInstr && (BC.MIB->isUnconditionalBranch(*LastInstr) ||
                        BC.MIB->isIndirectBranch(*LastInstr)))
        continue;
      // If there is an FT it will be the last successor.
      auto &SuccBI = *BB->branch_info_rbegin();
      auto &Succ = *BB->succ_rbegin();
      if (SuccBI.Count == 0) {
        SuccBI.Count = Inferred;
        SuccBI.MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
        Succ->ExecutionCount += Inferred;
      }
    }
  }
}

void BinaryFunction::clearProfile() {
  // Keep function execution profile the same. Only clear basic block and edge
  // counts.
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BB->ExecutionCount = 0;
    for (BinaryBasicBlock::BinaryBranchInfo &BI : BB->branch_info()) {
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}

} // namespace bolt
} // namespace llvm
