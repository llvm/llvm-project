//===- bolt/Passes/StackAllocationAnalysis.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_STACKALLOCATIONANALYSIS_H
#define BOLT_PASSES_STACKALLOCATIONANALYSIS_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {
class StackPointerTracking;

/// Perform a dataflow analysis to track the value of SP as an offset relative
/// to the CFA.
class StackAllocationAnalysis
    : public InstrsDataflowAnalysis<StackAllocationAnalysis,
                                    /*Backward=*/false> {
  friend class DataflowAnalysis<StackAllocationAnalysis, BitVector>;

  StackPointerTracking &SPT;

public:
  StackAllocationAnalysis(BinaryFunction &BF, StackPointerTracking &SPT,
                          MCPlusBuilder::AllocatorIdTy AllocId)
      : InstrsDataflowAnalysis<StackAllocationAnalysis, false>(BF, AllocId),
        SPT(SPT) {}
  virtual ~StackAllocationAnalysis() {}

  void run() { InstrsDataflowAnalysis<StackAllocationAnalysis, false>::run(); }

protected:
  void preflight();

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB);

  BitVector getStartingStateAtPoint(const MCInst &Point);

  void doConfluence(BitVector &StateOut, const BitVector &StateIn);

  BitVector doKill(const MCInst &Point, const BitVector &StateIn,
                   int DeallocSize);

  void doConfluenceWithLP(BitVector &StateOut, const BitVector &StateIn,
                          const MCInst &Invoke);

  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const {
    return StringRef("StackAllocationAnalysis");
  }
};

} // end namespace bolt
} // end namespace llvm

#endif
