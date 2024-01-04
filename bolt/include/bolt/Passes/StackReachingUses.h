//===- bolt/Passes/StackReachingUses.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_STACKREACHINGUSES_H
#define BOLT_PASSES_STACKREACHINGUSES_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

class FrameAnalysis;
struct FrameIndexEntry;

class StackReachingUses
    : public InstrsDataflowAnalysis<StackReachingUses, /*Backward=*/true> {
  friend class DataflowAnalysis<StackReachingUses, BitVector, true>;

public:
  StackReachingUses(const FrameAnalysis &FA, BinaryFunction &BF,
                    MCPlusBuilder::AllocatorIdTy AllocId = 0)
      : InstrsDataflowAnalysis(BF, AllocId), FA(FA) {}
  virtual ~StackReachingUses() {}

  /// Return true if the stack position written by the store in \p StoreFIE was
  /// later consumed by a load to a different register (not the same one used in
  /// the store). Useful for identifying loads/stores of callee-saved regs.
  bool isLoadedInDifferentReg(const FrameIndexEntry &StoreFIE,
                              ExprIterator Candidates) const;

  /// Answer whether the stack position written by the store represented in
  /// \p StoreFIE is loaded from or consumed in any way. The set of all
  /// relevant expressions reaching this store should be in \p Candidates.
  /// If \p IncludelocalAccesses is false, we only consider whether there is
  /// a callee that consumes this stack position.
  bool isStoreUsed(const FrameIndexEntry &StoreFIE, ExprIterator Candidates,
                   bool IncludeLocalAccesses = true) const;

  void run() { InstrsDataflowAnalysis<StackReachingUses, true>::run(); }

protected:
  // Reference to the result of stack frame analysis
  const FrameAnalysis &FA;

  void preflight();

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return BitVector(NumInstrs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(NumInstrs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
  }

  // Define the function computing the kill set -- whether expression Y, a
  // tracked expression, will be considered to be dead after executing X.
  bool doesXKillsY(const MCInst *X, const MCInst *Y);
  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const { return StringRef("StackReachingUses"); }
};

} // end namespace bolt
} // end namespace llvm

#endif
