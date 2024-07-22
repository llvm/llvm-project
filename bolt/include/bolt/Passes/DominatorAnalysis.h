//===- bolt/Passes/DominatorAnalysis.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_DOMINATORANALYSIS_H
#define BOLT_PASSES_DOMINATORANALYSIS_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

/// The whole reason for running a dominator analysis at the instruction level
/// (that is much more expensive than at the BB level) is because of invoke
/// instructions that may cause early exits in the middle of the BB, making half
/// of the BB potentially dominate the landing pad but not instructions after
/// the invoke.
template <bool Backward = false>
class DominatorAnalysis
    : public InstrsDataflowAnalysis<DominatorAnalysis<Backward>, Backward> {
  friend class DataflowAnalysis<DominatorAnalysis<Backward>, BitVector,
                                Backward>;

public:
  DominatorAnalysis(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId)
      : InstrsDataflowAnalysis<DominatorAnalysis<Backward>, Backward>(BF,
                                                                      AllocId) {
  }
  virtual ~DominatorAnalysis() {}

  SmallSetVector<ProgramPoint, 4> getDominanceFrontierFor(const MCInst &Dom) {
    SmallSetVector<ProgramPoint, 4> Result;
    uint64_t DomIdx = this->ExprToIdx[&Dom];
    assert(!Backward && "Post-dom frontier not implemented");
    for (BinaryBasicBlock &BB : this->Func) {
      bool HasDominatedPred = false;
      bool HasNonDominatedPred = false;
      SmallSetVector<ProgramPoint, 4> Candidates;
      this->doForAllSuccsOrPreds(BB, [&](ProgramPoint P) {
        if ((*this->getStateAt(P))[DomIdx]) {
          Candidates.insert(P);
          HasDominatedPred = true;
          return;
        }
        HasNonDominatedPred = true;
      });
      if (HasDominatedPred && HasNonDominatedPred)
        Result.insert(Candidates.begin(), Candidates.end());
      if ((*this->getStateAt(ProgramPoint::getLastPointAt(BB)))[DomIdx] &&
          BB.succ_begin() == BB.succ_end())
        Result.insert(ProgramPoint::getLastPointAt(BB));
    }
    return Result;
  }

  bool doesADominateB(const MCInst &A, unsigned BIdx) {
    return this->count(BIdx, A);
  }

  bool doesADominateB(const MCInst &A, const MCInst &B) {
    return this->count(B, A);
  }

  bool doesADominateB(const MCInst &A, ProgramPoint B) {
    return this->count(B, A);
  }

  bool doesADominateB(ProgramPoint A, const MCInst &B) {
    if (A.isInst())
      return doesADominateB(*A.getInst(), B);

    // This analysis keep track of which instructions dominates another
    // instruction, it doesn't keep track of BBs. So we need a non-empty
    // BB if we want to know whether this BB dominates something.
    BinaryBasicBlock *BB = A.getBB();
    while (BB->size() == 0) {
      if (BB->succ_size() == 0)
        return false;
      assert(BB->succ_size() == 1);
      BB = *BB->succ_begin();
    }
    const MCInst &InstA = *BB->begin();
    return doesADominateB(InstA, B);
  }

  void doForAllDominators(const MCInst &Inst,
                          std::function<void(const MCInst &)> Task) {
    for (auto I = this->expr_begin(Inst), E = this->expr_end(); I != E; ++I)
      Task(**I);
  }

  void run() {
    InstrsDataflowAnalysis<DominatorAnalysis<Backward>, Backward>::run();
  }

private:
  void preflight() {
    // Populate our universe of tracked expressions with all instructions
    // except pseudos
    for (BinaryBasicBlock &BB : this->Func) {
      for (MCInst &Inst : BB) {
        this->Expressions.push_back(&Inst);
        this->ExprToIdx[&Inst] = this->NumInstrs++;
      }
    }
  }

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // Entry points start with empty set
    // All others start with the full set.
    if (!Backward && BB.pred_size() == 0 && BB.throw_size() == 0)
      return BitVector(this->NumInstrs, false);
    if (Backward && BB.succ_size() == 0)
      return BitVector(this->NumInstrs, false);
    return BitVector(this->NumInstrs, true);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(this->NumInstrs, true);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut &= StateIn;
  }

  BitVector computeNext(const MCInst &Point, const BitVector &Cur) {
    BitVector Next = Cur;
    // Gen
    if (!this->BC.MIB->isCFI(Point))
      Next.set(this->ExprToIdx[&Point]);
    return Next;
  }

  StringRef getAnnotationName() const {
    if (Backward)
      return StringRef("PostDominatorAnalysis");
    return StringRef("DominatorAnalysis");
  }
};

} // end namespace bolt
} // end namespace llvm

#endif
