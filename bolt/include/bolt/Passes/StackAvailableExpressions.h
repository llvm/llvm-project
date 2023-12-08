//===- bolt/Passes/StackAvailableExpressions.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_STACKAVAILABLEEXPRESSIONS_H
#define BOLT_PASSES_STACKAVAILABLEEXPRESSIONS_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

class FrameAnalysis;
class RegAnalysis;

class StackAvailableExpressions
    : public InstrsDataflowAnalysis<StackAvailableExpressions> {
  friend class DataflowAnalysis<StackAvailableExpressions, BitVector>;

public:
  StackAvailableExpressions(const RegAnalysis &RA, const FrameAnalysis &FA,
                            BinaryFunction &BF);
  virtual ~StackAvailableExpressions() {}

  void run() { InstrsDataflowAnalysis<StackAvailableExpressions>::run(); }

protected:
  const RegAnalysis &RA;
  const FrameAnalysis &FA;

  void preflight();
  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB);
  BitVector getStartingStateAtPoint(const MCInst &Point);
  void doConfluence(BitVector &StateOut, const BitVector &StateIn);
  /// Define the function computing the kill set -- whether expression Y, a
  /// tracked expression, will be considered to be dead after executing X.
  bool doesXKillsY(const MCInst *X, const MCInst *Y);
  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const {
    return StringRef("StackAvailableExpressions");
  }
};

} // namespace bolt
} // namespace llvm

#endif
