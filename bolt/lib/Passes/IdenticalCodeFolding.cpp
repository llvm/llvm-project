//===- bolt/Passes/IdenticalCodeFolding.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IdenticalCodeFolding class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/IdenticalCodeFolding.h"
#include "bolt/Core/ParallelUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Timer.h"
#include <atomic>
#include <map>
#include <set>
#include <unordered_map>

#define DEBUG_TYPE "bolt-icf"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool> UseDFS("icf-dfs",
                            cl::desc("use DFS ordering when using -icf option"),
                            cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
TimeICF("time-icf",
  cl::desc("time icf steps"),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));
} // namespace opts

namespace {
using JumpTable = bolt::JumpTable;

/// Compare two jump tables in 2 functions. The function relies on consistent
/// ordering of basic blocks in both binary functions (e.g. DFS).
bool equalJumpTables(const JumpTable &JumpTableA, const JumpTable &JumpTableB,
                     const BinaryFunction &FunctionA,
                     const BinaryFunction &FunctionB) {
  if (JumpTableA.EntrySize != JumpTableB.EntrySize)
    return false;

  if (JumpTableA.Type != JumpTableB.Type)
    return false;

  if (JumpTableA.getSize() != JumpTableB.getSize())
    return false;

  for (uint64_t Index = 0; Index < JumpTableA.Entries.size(); ++Index) {
    const MCSymbol *LabelA = JumpTableA.Entries[Index];
    const MCSymbol *LabelB = JumpTableB.Entries[Index];

    const BinaryBasicBlock *TargetA = FunctionA.getBasicBlockForLabel(LabelA);
    const BinaryBasicBlock *TargetB = FunctionB.getBasicBlockForLabel(LabelB);

    if (!TargetA || !TargetB) {
      assert((TargetA || LabelA == FunctionA.getFunctionEndLabel()) &&
             "no target basic block found");
      assert((TargetB || LabelB == FunctionB.getFunctionEndLabel()) &&
             "no target basic block found");

      if (TargetA != TargetB)
        return false;

      continue;
    }

    assert(TargetA && TargetB && "cannot locate target block(s)");

    if (TargetA->getLayoutIndex() != TargetB->getLayoutIndex())
      return false;
  }

  return true;
}

/// Helper function that compares an instruction of this function to the
/// given instruction of the given function. The functions should have
/// identical CFG.
template <class Compare>
bool isInstrEquivalentWith(const MCInst &InstA, const BinaryBasicBlock &BBA,
                           const MCInst &InstB, const BinaryBasicBlock &BBB,
                           Compare Comp) {
  if (InstA.getOpcode() != InstB.getOpcode())
    return false;

  const BinaryContext &BC = BBA.getFunction()->getBinaryContext();

  // In this function we check for special conditions:
  //
  //    * instructions with landing pads
  //
  // Most of the common cases should be handled by MCPlus::equals()
  // that compares regular instruction operands.
  //
  // NB: there's no need to compare jump table indirect jump instructions
  //     separately as jump tables are handled by comparing corresponding
  //     symbols.
  const Optional<MCPlus::MCLandingPad> EHInfoA = BC.MIB->getEHInfo(InstA);
  const Optional<MCPlus::MCLandingPad> EHInfoB = BC.MIB->getEHInfo(InstB);

  if (EHInfoA || EHInfoB) {
    if (!EHInfoA && (EHInfoB->first || EHInfoB->second))
      return false;

    if (!EHInfoB && (EHInfoA->first || EHInfoA->second))
      return false;

    if (EHInfoA && EHInfoB) {
      // Action indices should match.
      if (EHInfoA->second != EHInfoB->second)
        return false;

      if (!EHInfoA->first != !EHInfoB->first)
        return false;

      if (EHInfoA->first && EHInfoB->first) {
        const BinaryBasicBlock *LPA = BBA.getLandingPad(EHInfoA->first);
        const BinaryBasicBlock *LPB = BBB.getLandingPad(EHInfoB->first);
        assert(LPA && LPB && "cannot locate landing pad(s)");

        if (LPA->getLayoutIndex() != LPB->getLayoutIndex())
          return false;
      }
    }
  }

  return BC.MIB->equals(InstA, InstB, Comp);
}

/// Returns true if this function has identical code and CFG with
/// the given function \p BF.
///
/// If \p CongruentSymbols is set to true, then symbolic operands that reference
/// potentially identical but different functions are ignored during the
/// comparison.
bool isIdenticalWith(const BinaryFunction &A, const BinaryFunction &B,
                     bool CongruentSymbols) {
  assert(A.hasCFG() && B.hasCFG() && "both functions should have CFG");

  // Compare the two functions, one basic block at a time.
  // Currently we require two identical basic blocks to have identical
  // instruction sequences and the same index in their corresponding
  // functions. The latter is important for CFG equality.

  if (A.layout_size() != B.layout_size())
    return false;

  // Comparing multi-entry functions could be non-trivial.
  if (A.isMultiEntry() || B.isMultiEntry())
    return false;

  // Process both functions in either DFS or existing order.
  const BinaryFunction::BasicBlockOrderType &OrderA =
      opts::UseDFS ? A.dfs() : A.getLayout();
  const BinaryFunction::BasicBlockOrderType &OrderB =
      opts::UseDFS ? B.dfs() : B.getLayout();

  const BinaryContext &BC = A.getBinaryContext();

  auto BBI = OrderB.begin();
  for (const BinaryBasicBlock *BB : OrderA) {
    const BinaryBasicBlock *OtherBB = *BBI;

    if (BB->getLayoutIndex() != OtherBB->getLayoutIndex())
      return false;

    // Compare successor basic blocks.
    // NOTE: the comparison for jump tables is only partially verified here.
    if (BB->succ_size() != OtherBB->succ_size())
      return false;

    auto SuccBBI = OtherBB->succ_begin();
    for (const BinaryBasicBlock *SuccBB : BB->successors()) {
      const BinaryBasicBlock *SuccOtherBB = *SuccBBI;
      if (SuccBB->getLayoutIndex() != SuccOtherBB->getLayoutIndex())
        return false;
      ++SuccBBI;
    }

    // Compare all instructions including pseudos.
    auto I = BB->begin(), E = BB->end();
    auto OtherI = OtherBB->begin(), OtherE = OtherBB->end();
    while (I != E && OtherI != OtherE) {
      // Compare symbols.
      auto AreSymbolsIdentical = [&](const MCSymbol *SymbolA,
                                     const MCSymbol *SymbolB) {
        if (SymbolA == SymbolB)
          return true;

        // All local symbols are considered identical since they affect a
        // control flow and we check the control flow separately.
        // If a local symbol is escaped, then the function (potentially) has
        // multiple entry points and we exclude such functions from
        // comparison.
        if (SymbolA->isTemporary() && SymbolB->isTemporary())
          return true;

        // Compare symbols as functions.
        uint64_t EntryIDA = 0;
        uint64_t EntryIDB = 0;
        const BinaryFunction *FunctionA =
            BC.getFunctionForSymbol(SymbolA, &EntryIDA);
        const BinaryFunction *FunctionB =
            BC.getFunctionForSymbol(SymbolB, &EntryIDB);
        if (FunctionA && EntryIDA)
          FunctionA = nullptr;
        if (FunctionB && EntryIDB)
          FunctionB = nullptr;
        if (FunctionA && FunctionB) {
          // Self-referencing functions and recursive calls.
          if (FunctionA == &A && FunctionB == &B)
            return true;

          // Functions with different hash values can never become identical,
          // hence A and B are different.
          if (CongruentSymbols)
            return FunctionA->getHash() == FunctionB->getHash();

          return FunctionA == FunctionB;
        }

        // One of the symbols represents a function, the other one does not.
        if (FunctionA != FunctionB)
          return false;

        // Check if symbols are jump tables.
        const BinaryData *SIA = BC.getBinaryDataByName(SymbolA->getName());
        if (!SIA)
          return false;
        const BinaryData *SIB = BC.getBinaryDataByName(SymbolB->getName());
        if (!SIB)
          return false;

        assert((SIA->getAddress() != SIB->getAddress()) &&
               "different symbols should not have the same value");

        const JumpTable *JumpTableA =
            A.getJumpTableContainingAddress(SIA->getAddress());
        if (!JumpTableA)
          return false;

        const JumpTable *JumpTableB =
            B.getJumpTableContainingAddress(SIB->getAddress());
        if (!JumpTableB)
          return false;

        if ((SIA->getAddress() - JumpTableA->getAddress()) !=
            (SIB->getAddress() - JumpTableB->getAddress()))
          return false;

        return equalJumpTables(*JumpTableA, *JumpTableB, A, B);
      };

      if (!isInstrEquivalentWith(*I, *BB, *OtherI, *OtherBB,
                                 AreSymbolsIdentical))
        return false;

      ++I;
      ++OtherI;
    }

    // One of the identical blocks may have a trailing unconditional jump that
    // is ignored for CFG purposes.
    const MCInst *TrailingInstr =
        (I != E ? &(*I) : (OtherI != OtherE ? &(*OtherI) : 0));
    if (TrailingInstr && !BC.MIB->isUnconditionalBranch(*TrailingInstr))
      return false;

    ++BBI;
  }

  // Compare exceptions action tables.
  if (A.getLSDAActionTable() != B.getLSDAActionTable() ||
      A.getLSDATypeTable() != B.getLSDATypeTable() ||
      A.getLSDATypeIndexTable() != B.getLSDATypeIndexTable())
    return false;

  return true;
}

// This hash table is used to identify identical functions. It maps
// a function to a bucket of functions identical to it.
struct KeyHash {
  size_t operator()(const BinaryFunction *F) const { return F->getHash(); }
};

/// Identify two congruent functions. Two functions are considered congruent,
/// if they are identical/equal except for some of their instruction operands
/// that reference potentially identical functions, i.e. functions that could
/// be folded later. Congruent functions are candidates for folding in our
/// iterative ICF algorithm.
///
/// Congruent functions are required to have identical hash.
struct KeyCongruent {
  bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
    if (A == B)
      return true;
    return isIdenticalWith(*A, *B, /*CongruentSymbols=*/true);
  }
};

struct KeyEqual {
  bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
    if (A == B)
      return true;
    return isIdenticalWith(*A, *B, /*CongruentSymbols=*/false);
  }
};

typedef std::unordered_map<BinaryFunction *, std::set<BinaryFunction *>,
                           KeyHash, KeyCongruent>
    CongruentBucketsMap;

typedef std::unordered_map<BinaryFunction *, std::vector<BinaryFunction *>,
                           KeyHash, KeyEqual>
    IdenticalBucketsMap;

std::string hashInteger(uint64_t Value) {
  std::string HashString;
  if (Value == 0)
    HashString.push_back(0);

  while (Value) {
    uint8_t LSB = Value & 0xff;
    HashString.push_back(LSB);
    Value >>= 8;
  }

  return HashString;
}

std::string hashSymbol(BinaryContext &BC, const MCSymbol &Symbol) {
  std::string HashString;

  // Ignore function references.
  if (BC.getFunctionForSymbol(&Symbol))
    return HashString;

  llvm::ErrorOr<uint64_t> ErrorOrValue = BC.getSymbolValue(Symbol);
  if (!ErrorOrValue)
    return HashString;

  // Ignore jump table references.
  if (BC.getJumpTableContainingAddress(*ErrorOrValue))
    return HashString;

  return HashString.append(hashInteger(*ErrorOrValue));
}

std::string hashExpr(BinaryContext &BC, const MCExpr &Expr) {
  switch (Expr.getKind()) {
  case MCExpr::Constant:
    return hashInteger(cast<MCConstantExpr>(Expr).getValue());
  case MCExpr::SymbolRef:
    return hashSymbol(BC, cast<MCSymbolRefExpr>(Expr).getSymbol());
  case MCExpr::Unary: {
    const auto &UnaryExpr = cast<MCUnaryExpr>(Expr);
    return hashInteger(UnaryExpr.getOpcode())
        .append(hashExpr(BC, *UnaryExpr.getSubExpr()));
  }
  case MCExpr::Binary: {
    const auto &BinaryExpr = cast<MCBinaryExpr>(Expr);
    return hashExpr(BC, *BinaryExpr.getLHS())
        .append(hashInteger(BinaryExpr.getOpcode()))
        .append(hashExpr(BC, *BinaryExpr.getRHS()));
  }
  case MCExpr::Target:
    return std::string();
  }

  llvm_unreachable("invalid expression kind");
}

std::string hashInstOperand(BinaryContext &BC, const MCOperand &Operand) {
  if (Operand.isImm())
    return hashInteger(Operand.getImm());
  if (Operand.isReg())
    return hashInteger(Operand.getReg());
  if (Operand.isExpr())
    return hashExpr(BC, *Operand.getExpr());

  return std::string();
}

} // namespace

namespace llvm {
namespace bolt {

void IdenticalCodeFolding::runOnFunctions(BinaryContext &BC) {
  const size_t OriginalFunctionCount = BC.getBinaryFunctions().size();
  uint64_t NumFunctionsFolded = 0;
  std::atomic<uint64_t> NumJTFunctionsFolded{0};
  std::atomic<uint64_t> BytesSavedEstimate{0};
  std::atomic<uint64_t> CallsSavedEstimate{0};
  std::atomic<uint64_t> NumFoldedLastIteration{0};
  CongruentBucketsMap CongruentBuckets;

  // Hash all the functions
  auto hashFunctions = [&]() {
    NamedRegionTimer HashFunctionsTimer("hashing", "hashing", "ICF breakdown",
                                        "ICF breakdown", opts::TimeICF);
    ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
      // Make sure indices are in-order.
      BF.updateLayoutIndices();

      // Pre-compute hash before pushing into hashtable.
      // Hash instruction operands to minimize hash collisions.
      BF.computeHash(opts::UseDFS, [&BC](const MCOperand &Op) {
        return hashInstOperand(BC, Op);
      });
    };

    ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
      return !shouldOptimize(BF);
    };

    ParallelUtilities::runOnEachFunction(
        BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, SkipFunc,
        "hashFunctions", /*ForceSequential*/ false, 2);
  };

  // Creates buckets with congruent functions - functions that potentially
  // could  be folded.
  auto createCongruentBuckets = [&]() {
    NamedRegionTimer CongruentBucketsTimer("congruent buckets",
                                           "congruent buckets", "ICF breakdown",
                                           "ICF breakdown", opts::TimeICF);
    for (auto &BFI : BC.getBinaryFunctions()) {
      BinaryFunction &BF = BFI.second;
      if (!this->shouldOptimize(BF))
        continue;
      CongruentBuckets[&BF].emplace(&BF);
    }
  };

  // Partition each set of congruent functions into sets of identical functions
  // and fold them
  auto performFoldingPass = [&]() {
    NamedRegionTimer FoldingPassesTimer("folding passes", "folding passes",
                                        "ICF breakdown", "ICF breakdown",
                                        opts::TimeICF);
    Timer SinglePass("single fold pass", "single fold pass");
    LLVM_DEBUG(SinglePass.startTimer());

    ThreadPool *ThPool;
    if (!opts::NoThreads)
      ThPool = &ParallelUtilities::getThreadPool();

    // Fold identical functions within a single congruent bucket
    auto processSingleBucket = [&](std::set<BinaryFunction *> &Candidates) {
      Timer T("folding single congruent list", "folding single congruent list");
      LLVM_DEBUG(T.startTimer());

      // Identical functions go into the same bucket.
      IdenticalBucketsMap IdenticalBuckets;
      for (BinaryFunction *BF : Candidates) {
        IdenticalBuckets[BF].emplace_back(BF);
      }

      for (auto &IBI : IdenticalBuckets) {
        // Functions identified as identical.
        std::vector<BinaryFunction *> &Twins = IBI.second;
        if (Twins.size() < 2)
          continue;

        // Fold functions. Keep the order consistent across invocations with
        // different options.
        llvm::stable_sort(
            Twins, [](const BinaryFunction *A, const BinaryFunction *B) {
              return A->getFunctionNumber() < B->getFunctionNumber();
            });

        BinaryFunction *ParentBF = Twins[0];
        for (unsigned I = 1; I < Twins.size(); ++I) {
          BinaryFunction *ChildBF = Twins[I];
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: folding " << *ChildBF << " into "
                            << *ParentBF << '\n');

          // Remove child function from the list of candidates.
          auto FI = Candidates.find(ChildBF);
          assert(FI != Candidates.end() &&
                 "function expected to be in the set");
          Candidates.erase(FI);

          // Fold the function and remove from the list of processed functions.
          BytesSavedEstimate += ChildBF->getSize();
          CallsSavedEstimate += std::min(ChildBF->getKnownExecutionCount(),
                                         ParentBF->getKnownExecutionCount());
          BC.foldFunction(*ChildBF, *ParentBF);

          ++NumFoldedLastIteration;

          if (ParentBF->hasJumpTables())
            ++NumJTFunctionsFolded;
        }
      }

      LLVM_DEBUG(T.stopTimer());
    };

    // Create a task for each congruent bucket
    for (auto &Entry : CongruentBuckets) {
      std::set<BinaryFunction *> &Bucket = Entry.second;
      if (Bucket.size() < 2)
        continue;

      if (opts::NoThreads)
        processSingleBucket(Bucket);
      else
        ThPool->async(processSingleBucket, std::ref(Bucket));
    }

    if (!opts::NoThreads)
      ThPool->wait();

    LLVM_DEBUG(SinglePass.stopTimer());
  };

  hashFunctions();
  createCongruentBuckets();

  unsigned Iteration = 1;
  // We repeat the pass until no new modifications happen.
  do {
    NumFoldedLastIteration = 0;
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ICF iteration " << Iteration << "...\n");

    performFoldingPass();

    NumFunctionsFolded += NumFoldedLastIteration;
    ++Iteration;

  } while (NumFoldedLastIteration > 0);

  LLVM_DEBUG({
    // Print functions that are congruent but not identical.
    for (auto &CBI : CongruentBuckets) {
      std::set<BinaryFunction *> &Candidates = CBI.second;
      if (Candidates.size() < 2)
        continue;
      dbgs() << "BOLT-DEBUG: the following " << Candidates.size()
             << " functions (each of size " << (*Candidates.begin())->getSize()
             << " bytes) are congruent but not identical:\n";
      for (BinaryFunction *BF : Candidates) {
        dbgs() << "  " << *BF;
        if (BF->getKnownExecutionCount())
          dbgs() << " (executed " << BF->getKnownExecutionCount() << " times)";
        dbgs() << '\n';
      }
    }
  });

  if (NumFunctionsFolded)
    outs() << "BOLT-INFO: ICF folded " << NumFunctionsFolded << " out of "
           << OriginalFunctionCount << " functions in " << Iteration
           << " passes. " << NumJTFunctionsFolded
           << " functions had jump tables.\n"
           << "BOLT-INFO: Removing all identical functions will save "
           << format("%.2lf", (double)BytesSavedEstimate / 1024)
           << " KB of code space. Folded functions were called "
           << CallsSavedEstimate << " times based on profile.\n";
}

} // namespace bolt
} // namespace llvm
