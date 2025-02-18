//===- bolt/Passes/RedirectNeverTakenJumps.cpp - Code size reduction ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements RedirectNeverTakenJumps class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RedirectNeverTakenJumps.h"
#include "bolt/Core/ParallelUtilities.h"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool> RedirectNeverTakenJumps(
    "redirect-never-taken-jumps",
    cl::desc("Apply a heuristic to redirect never taken jumps in order to "
             "reduce hot code size (X86 only)"),
    cl::Hidden, cl::init(false), cl::cat(BoltOptCategory));

static cl::opt<bool> AggressiveNeverTaken(
    "aggressive-never-taken",
    cl::desc("Classify all zero-execution-count jumps as never taken. This "
             "option ignores the possibility of execution counts of hot jumps "
             "being incorrectly set to 0 in the input profile"),
    cl::ReallyHidden, cl::init(false), cl::cat(BoltOptCategory));

static cl::opt<double> ConservativeNeverTakenThreshold(
    "conservative-never-taken-threshold",
    cl::desc(
        "When aggressive-never-taken=0 (default), this value controls how "
        "conservative the classification of never-taken jumps is. The smaller "
        "the value the more conservative the classification. In most realistic "
        "settings, the value should exceed 1.0. Default 1.25."),
    cl::ZeroOrMore, cl::init(1.25), cl::ReallyHidden, cl::cat(BoltOptCategory));
} // namespace opts

namespace {
/// A jump instruction in the binary.
struct JumpT {
  JumpT(const JumpT &) = delete;
  JumpT(JumpT &&) = default;
  JumpT &operator=(const JumpT &) = delete;
  JumpT &operator=(JumpT &&) = default;

  explicit JumpT(MCInst *Inst, unsigned CC, bool IsUnconditional,
                 BinaryBasicBlock *OriginalTargetBB, uint64_t ExecutionCount,
                 BinaryBasicBlock *HomeBB, uint64_t OriginalAddress,
                 uint64_t OriginalInstrSize)
      : Inst(Inst), CC(CC), IsUnconditional(IsUnconditional),
        OriginalTargetBB(OriginalTargetBB), ExecutionCount(ExecutionCount),
        HomeBB(HomeBB), OriginalAddress(OriginalAddress),
        OriginalInstrSize(OriginalInstrSize) {}

  MCInst *Inst;
  unsigned CC;
  bool IsUnconditional;
  BinaryBasicBlock *OriginalTargetBB;
  uint64_t ExecutionCount;
  BinaryBasicBlock *HomeBB;
  uint64_t OriginalAddress{0};
  uint8_t OriginalInstrSize{0};

  bool IsLongNeverTaken{false};
  bool IsRedirectionTarget{false};
  JumpT *RedirectionTarget{nullptr};
  JumpT *UncondJumpInSameBlock{nullptr};
};

using Jumps = std::vector<std::unique_ptr<JumpT>>;
using JumpPtrs = std::vector<JumpT *>;
using FlowMapTy = std::unordered_map<const BinaryBasicBlock *, uint64_t>;
using BlockToJumpsMapTy =
    std::unordered_map<BinaryBasicBlock *, std::vector<JumpT *>>;

/// Size of jump instructions in bytes in X86.
static constexpr uint8_t ShortJumpSize = 2;
static constexpr uint8_t LongUncondJumpSize = 5;
static constexpr uint8_t LongCondJumpSize = 6;

/// The longest distance for any short jump on X86.
static constexpr uint8_t ShortJumpBits = 8;
static constexpr uint8_t ShortestJumpSpan = 1ULL << (ShortJumpBits - 1);

bool isLongJump(const uint64_t JumpStartAddr, const uint64_t JumpEndAddr,
                const bool SameFragment) {
  if (!SameFragment)
    return true;
  if (JumpEndAddr > JumpStartAddr)
    return JumpEndAddr - JumpStartAddr > ShortestJumpSpan - 1;
  else
    return JumpStartAddr - JumpEndAddr > ShortestJumpSpan;
}

void createJumps(BinaryFunction &Function, FunctionFragment &Fragment,
                 Jumps &JumpsInFunction, JumpPtrs &JumpsInFragment) {
  const BinaryContext &BC = Function.getBinaryContext();

  auto createJump = [&](MCInst *Branch, bool IsUnconditional,
                        BinaryBasicBlock *SourceBB, BinaryBasicBlock *TargetBB,
                        const uint8_t OffsetFromBlockEnd) {
    const BinaryBasicBlock::BinaryBranchInfo &BI =
        SourceBB->getBranchInfo(*TargetBB);
    uint64_t ExecCount = 0;
    if (BI.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
      ExecCount = BI.Count;

    const uint64_t JumpEndAddr = TargetBB->getOutputAddressRange().first;
    const uint64_t JumpStartAddr =
        SourceBB->getOutputAddressRange().second - OffsetFromBlockEnd;
    const uint8_t LongJumpSize =
        IsUnconditional ? LongUncondJumpSize : LongCondJumpSize;
    const uint8_t JumpInstrSize =
        isLongJump(JumpStartAddr, JumpEndAddr,
                   SourceBB->getFragmentNum() == TargetBB->getFragmentNum())
            ? LongJumpSize
            : ShortJumpSize;
    return std::unique_ptr<JumpT>(new JumpT(
        Branch, BC.MIB->getCondCode(*Branch), IsUnconditional, TargetBB,
        ExecCount, SourceBB, JumpStartAddr - JumpInstrSize, JumpInstrSize));
  };

  for (BinaryBasicBlock *BB : Fragment) {
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    BinaryBasicBlock *CondSuccessor = nullptr;
    BinaryBasicBlock *UncondSuccessor = nullptr;

    if (BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
      if (BB->succ_size() == 1) {
        UncondSuccessor = BB->getSuccessor();
        if (UncondBranch != nullptr) {
          std::unique_ptr<JumpT> Jump =
              createJump(UncondBranch, true, BB, UncondSuccessor, 0);
          JumpsInFragment.push_back(Jump.get());
          JumpsInFunction.push_back(std::move(Jump));
        }
      } else if (BB->succ_size() == 2) {
        assert(CondBranch != nullptr);
        CondSuccessor = BB->getConditionalSuccessor(true);
        UncondSuccessor = BB->getConditionalSuccessor(false);
        std::unique_ptr<JumpT> UncondJump = nullptr;
        std::unique_ptr<JumpT> CondJump = nullptr;
        uint8_t UncondJumpInstrSize = 0;
        if (UncondBranch != nullptr) {
          UncondJump = createJump(UncondBranch, true, BB, UncondSuccessor, 0);
          UncondJumpInstrSize = UncondJump->OriginalInstrSize;
        }
        if (!BC.MIB->isDynamicBranch(*CondBranch)) {
          CondJump = createJump(CondBranch, false, BB, CondSuccessor,
                                UncondJumpInstrSize);
          if (UncondJump != nullptr)
            CondJump->UncondJumpInSameBlock = UncondJump.get();
        }
        if (CondJump != nullptr) {
          JumpsInFragment.push_back(CondJump.get());
          JumpsInFunction.push_back(std::move(CondJump));
        }
        if (UncondJump != nullptr) {
          JumpsInFragment.push_back(UncondJump.get());
          JumpsInFunction.push_back(std::move(UncondJump));
        }
      }
    }
  }
}

void identifyCandidates(BinaryFunction &Function, JumpPtrs &JumpsInFragment,
                        BlockToJumpsMapTy &TargetsToJumps) {
  // Identify jumps that are long and never taken.
  // First check if each jump is long and have zero execution count.
  auto isLongZeroCount = [&](const JumpT &Jump) {
    return Jump.ExecutionCount == 0 && Jump.OriginalInstrSize > ShortJumpSize;
    ;
  };

  BlockToJumpsMapTy SourcesToJumps;
  for (JumpT *Jump : JumpsInFragment) {
    Jump->IsLongNeverTaken = isLongZeroCount(*Jump);
    assert(Jump->OriginalTargetBB != nullptr);
    TargetsToJumps[Jump->OriginalTargetBB].push_back(Jump);
    SourcesToJumps[Jump->HomeBB].push_back(Jump);
  }

  // Next identify zero-execution-count jumps that are unlikely to actually be
  // never-taken by comparing the value of inflow (resp outflow) of each basic
  // block with its block execution count.
  FlowMapTy IncomingMap;
  FlowMapTy OutgoingMap;
  for (const BinaryBasicBlock &BB : Function) {
    auto SuccBIIter = BB.branch_info_begin();
    for (BinaryBasicBlock *Succ : BB.successors()) {
      const uint64_t Count = SuccBIIter->Count;
      if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
        ++SuccBIIter;
        continue;
      }
      IncomingMap[Succ] += Count;
      OutgoingMap[&BB] += Count;
      ++SuccBIIter;
    }
  }

  if (!opts::AggressiveNeverTaken) {
    for (auto &TargetToJumps : TargetsToJumps) {
      const BinaryBasicBlock *TargetBB = TargetToJumps.first;
      if (TargetBB->getKnownExecutionCount() == 0)
        continue;
      const uint64_t IncomingCount = IncomingMap[TargetBB];
      // If there is a noticeable gap between the incoming edge count and the BB
      // execution count, then we don't want to trust the 0 execution count
      // edges as actually 0 execution count.
      if (IncomingCount * opts::ConservativeNeverTakenThreshold <
          TargetBB->getKnownExecutionCount()) {
        for (JumpT *Jump : TargetToJumps.second) {
          Jump->IsLongNeverTaken = false;
        }
      }
    }

    for (auto &SourceToJumps : SourcesToJumps) {
      const BinaryBasicBlock *SourceBB = SourceToJumps.first;
      if (SourceBB->getKnownExecutionCount() == 0)
        continue;
      const uint64_t OutgoingCount = OutgoingMap[SourceBB];
      // If there is a noticeable gap between the outgoing edge count and the BB
      // execution count, then we don't want to trust the 0 execution count
      // edges as actually 0 execution count.

      if (OutgoingCount * opts::ConservativeNeverTakenThreshold <
          SourceBB->getKnownExecutionCount()) {
        for (JumpT *Jump : SourceToJumps.second) {
          Jump->IsLongNeverTaken = false;
        }
      }
    }
  }
}

uint64_t makeRedirectionDecisions(BlockToJumpsMapTy &TargetsToJumps) {
  uint64_t NumRedirected = 0;
  for (auto &TargetToJumps : TargetsToJumps) {
    std::vector<JumpT *> &Jumps = TargetToJumps.second;
    if (Jumps.size() <= 1)
      continue;
    std::unordered_map<unsigned, JumpT *> MostRecentCondJumps;
    JumpT *MostRecentUncondJump = nullptr;

    // Round 1: redirect jumps to the closest candidate to its right.
    for (auto JumpItr = Jumps.rbegin(); JumpItr != Jumps.rend(); ++JumpItr) {
      JumpT *CurrJump = *JumpItr;
      if (CurrJump->IsLongNeverTaken) {
        // Check if we can redirect CurrJump to MostRecentUncondJump.
        if (MostRecentUncondJump != nullptr) {
          if (!isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                          MostRecentUncondJump->OriginalAddress, true)) {
            // Redirect CurrJump to MostRecentUncondJump if the latter is close
            // enough.
            CurrJump->RedirectionTarget = MostRecentUncondJump;
            MostRecentUncondJump->IsRedirectionTarget = true;
            NumRedirected++;
          } else if (!CurrJump->IsUnconditional) {
            // Otherwise, try to redirect CurrJump to the most recent
            // conditional jump with the same conditional code.
            JumpT *MostRecentCondJump = MostRecentCondJumps[CurrJump->CC];
            if (MostRecentCondJump != nullptr &&
                !isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                            MostRecentCondJump->OriginalAddress, true)) {
              CurrJump->RedirectionTarget = MostRecentCondJump;
              MostRecentCondJump->IsRedirectionTarget = true;
              NumRedirected++;
            }
          }
        } else if (!CurrJump->IsUnconditional) {
          // If MostRecentUncondJump does not exist and CurrJump is conditional,
          // try to redirect CurrJump to the most recent conditional jump with
          // the same conditional code
          JumpT *MostRecentCondJump = MostRecentCondJumps[CurrJump->CC];
          if (MostRecentCondJump != nullptr &&
              !isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                          MostRecentCondJump->OriginalAddress, true)) {
            CurrJump->RedirectionTarget = MostRecentCondJump;
            MostRecentCondJump->IsRedirectionTarget = true;
            NumRedirected++;
          }
        }
      }

      // Update most recent jump by condition.
      if (CurrJump->IsUnconditional)
        MostRecentUncondJump = CurrJump;
      else
        MostRecentCondJumps[CurrJump->CC] = CurrJump;
    }

    // Round 2: redirect jumps to the closest candidate to its left while
    // making shre there are no cyclic redirections.
    MostRecentCondJumps.clear();
    MostRecentUncondJump = nullptr;
    for (auto JumpItr = Jumps.begin(); JumpItr != Jumps.end(); ++JumpItr) {
      JumpT *CurrJump = *JumpItr;
      if (CurrJump->IsLongNeverTaken) {
        if (CurrJump->RedirectionTarget == nullptr) {
          // Check if we can redirect CurrJump to MostRecentUncondJump.
          if (MostRecentUncondJump != nullptr) {
            if (!isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                            MostRecentUncondJump->OriginalAddress, true)) {
              // Redirect CurrJump to MostRecentUncondJump if the latter is
              // close enough.
              CurrJump->RedirectionTarget = MostRecentUncondJump;
              MostRecentUncondJump->IsRedirectionTarget = true;
              NumRedirected++;
            } else if (!CurrJump->IsUnconditional) {
              // Otherwise, try to redirect CurrJump to the most recent
              // conditional jump with the same conditional code.
              JumpT *MostRecentCondJump = MostRecentCondJumps[CurrJump->CC];
              if (MostRecentCondJump != nullptr &&
                  !isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                              MostRecentCondJump->OriginalAddress, true)) {
                CurrJump->RedirectionTarget = MostRecentCondJump;
                MostRecentCondJump->IsRedirectionTarget = true;
                NumRedirected++;
              }
            }
          } else if (!CurrJump->IsUnconditional) {
            // If MostRecentUncondJump does not exist and CurrJump is
            // conditional, try to redirect CurrJump to the most recent
            // conditional jump with the same conditional code
            JumpT *MostRecentCondJump = MostRecentCondJumps[CurrJump->CC];
            if (MostRecentCondJump != nullptr &&
                !isLongJump(CurrJump->OriginalAddress + ShortJumpSize,
                            MostRecentCondJump->OriginalAddress, true)) {
              CurrJump->RedirectionTarget = MostRecentCondJump;
              MostRecentCondJump->IsRedirectionTarget = true;
              NumRedirected++;
            }
          }
        } else {
          // If CurrJump has already been redirected in round 1, then use
          // continue to avoid updating MostRecentUncondJump or
          // MostRecentCondJumps with CurrJump. This will disallow redirection
          // to jumps that were redirected in round 1 and hence avoid cyclic
          // redirections.
          continue;
        }
      }

      // Update most recent jump by condition.
      if (CurrJump->IsUnconditional)
        MostRecentUncondJump = CurrJump;
      else
        MostRecentCondJumps[CurrJump->CC] = CurrJump;
    }
  }
  return NumRedirected;
}

void checkDecisionCorrectness(Jumps &JumpsInFunction) {
  // Check correctness of redirection decisions.
  for (const auto &Jump : JumpsInFunction) {
    if (Jump->RedirectionTarget != nullptr) {
      JumpT *CurrJump = Jump.get();
      JumpT *NextJump = CurrJump->RedirectionTarget;
      while (NextJump != nullptr) {
        // No cyclic redirections.
        assert(NextJump != Jump.get());
        // Redirect either to unconditional jump or jump with the same
        // conditional code.
        assert(NextJump->CC == CurrJump->CC || NextJump->IsUnconditional);
        CurrJump = NextJump;
        NextJump = CurrJump->RedirectionTarget;
      }
      // Jump will eventually reach its original target.
      assert(CurrJump->OriginalTargetBB == Jump->OriginalTargetBB);
    }
  }
}

void redirectJumps(BinaryFunction &Function, Jumps &JumpsInFunction) {
  const BinaryContext &BC = Function.getBinaryContext();
  // Helper function to split HomeBB at the JumpInst and return the new
  // basic block that has JumpInst as its first instruction.
  auto createJumpBlock = [&](JumpT *Jump) {
    BinaryBasicBlock *HomeBB = Jump->HomeBB;
    MCInst *Inst = Jump->Inst;

    // Obtain iterator II pointing to Inst.
    auto II = HomeBB->end();
    while (&*II != Inst)
      II--;
    return HomeBB->splitInPlaceAt(II);
  };

  // Split basic blocks at jump instructions that are redirection targets.
  for (auto JumpItr = JumpsInFunction.rbegin();
       JumpItr != JumpsInFunction.rend(); ++JumpItr) {
    JumpT *Jump = (*JumpItr).get();
    if (!Jump->IsRedirectionTarget)
      continue;
    BinaryBasicBlock *NewBB = createJumpBlock(Jump);
    Jump->HomeBB = NewBB;
    // If the new block contains two instructions, then it means NewBB
    // contains both a conditional jump (Jump) and an unconditional
    // jump (Jump->UncondJumpInSameBlock). We also need to update
    // the HomeBB of the latter.
    if (NewBB->getNumNonPseudos() == 2) {
      assert(Jump->UncondJumpInSameBlock != nullptr);
      Jump->UncondJumpInSameBlock->HomeBB = NewBB;
    }
  }

  // Check correctness of splitting.
  for (const auto &Jump : JumpsInFunction) {
    if (Jump->IsRedirectionTarget) {
      MCInst FirstInst = *(Jump->HomeBB->begin());
      assert(BC.MIB->getCondCode(FirstInst) == Jump->CC);
      assert(BC.MIB->getTargetSymbol(FirstInst) ==
             Jump->OriginalTargetBB->getLabel());
    }
  }

  // Perform redirections.
  for (const auto &Jump : JumpsInFunction) {
    if (Jump->RedirectionTarget != nullptr) {
      BinaryBasicBlock *HomeBB = Jump->HomeBB;
      BinaryBasicBlock *OriginalTargetBB = Jump->OriginalTargetBB;
      BinaryBasicBlock *NewTargetBB = Jump->RedirectionTarget->HomeBB;
      HomeBB->replaceSuccessor(OriginalTargetBB, NewTargetBB, /*Count=*/0,
                               /*MispredictedCount=*/0);
    }
  }
}
} // namespace

void RedirectNeverTakenJumps::performRedirections(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();

  // Populate BinaryBasicBlock::OutputAddressRange.
  uint64_t OldHotSize = 0;
  uint64_t OldColdSize = 0;
  std::tie(OldHotSize, OldColdSize) =
      BC.calculateEmittedSize(Function, /*FixBranches=*/true);

  // Perform redirections.
  Jumps JumpsInFunction;
  uint64_t NumJumpsToRedirect = 0;
  for (FunctionFragment &FF : Function.getLayout().fragments()) {
    JumpPtrs JumpsInFragment;
    BlockToJumpsMapTy TargetsToJumps;
    createJumps(Function, FF, JumpsInFunction, JumpsInFragment);
    identifyCandidates(Function, JumpsInFragment, TargetsToJumps);
    NumJumpsToRedirect += makeRedirectionDecisions(TargetsToJumps);
  }
  if (NumJumpsToRedirect == 0)
    return;

  checkDecisionCorrectness(JumpsInFunction);
  redirectJumps(Function, JumpsInFunction);

  // Log size reduction.
  const auto [NewHotSize, NewColdSize] =
      BC.calculateEmittedSize(Function, /*FixBranches*/ true);

  assert(NewHotSize <= OldHotSize);
  assert(NewColdSize <= OldColdSize);

  TotalSizeSavings += OldHotSize - NewHotSize + OldColdSize - NewColdSize;
  if (Function.hasValidIndex())
    TotalHotSizeSavings += OldHotSize - NewHotSize;
  return;
}

Error RedirectNeverTakenJumps::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return Error::success();

  if (opts::RedirectNeverTakenJumps) {
    ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
      return !shouldOptimize(BF);
    };

    ParallelUtilities::runOnEachFunction(
        BC, ParallelUtilities::SchedulingPolicy::SP_BB_LINEAR,
        [&](BinaryFunction &BF) { performRedirections(BF); }, SkipFunc,
        "RedirectNeverTakenJumps", /*ForceSequential*/ false);

    BC.outs() << format(
        "BOLT-INFO: redirection of never-taken jumps saved %zu bytes hot "
        "section code size and %zu bytes total code size\n",
        TotalHotSizeSavings.load(), TotalSizeSavings.load());
  }

  return Error::success();
}
