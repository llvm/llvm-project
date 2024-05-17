//===- bolt/Passes/ReachingInsns.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REACHINGINSNS_H
#define BOLT_PASSES_REACHINGINSNS_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

template <bool Backward = false>
class ReachingInsns
    : public InstrsDataflowAnalysis<ReachingInsns<Backward>, Backward> {
  friend class DataflowAnalysis<ReachingInsns<Backward>, BitVector, Backward>;

public:
  ReachingInsns(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId = 0)
      : InstrsDataflowAnalysis<ReachingInsns, Backward>(BF, AllocId) {}
  virtual ~ReachingInsns() {}

  bool isInLoop(const BinaryBasicBlock &BB) {
    const MCInst *First = BB.begin() != BB.end() ? &*BB.begin() : nullptr;
    assert(First && "This analysis does not work for empty BB");
    return ((*this->getStateAt(BB))[this->ExprToIdx[First]]);
  }

  bool isInLoop(const MCInst &Inst) {
    const BinaryBasicBlock *BB = InsnToBB[&Inst];
    assert(BB && "Unknown instruction");
    return isInLoop(*BB);
  }

  void run() {
    InstrsDataflowAnalysis<ReachingInsns<Backward>, Backward>::run();
  }

protected:
  std::unordered_map<const MCInst *, BinaryBasicBlock *> InsnToBB;

  void preflight() {
    for (BinaryBasicBlock &BB : this->Func) {
      for (MCInst &Inst : BB) {
        this->Expressions.push_back(&Inst);
        this->ExprToIdx[&Inst] = this->NumInstrs++;
        InsnToBB[&Inst] = &BB;
      }
    }
  }

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return BitVector(this->NumInstrs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(this->NumInstrs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
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
      return StringRef("ReachingInsnsBackward");
    return StringRef("ReachingInsns");
  }
};

} // end namespace bolt
} // end namespace llvm

#endif
