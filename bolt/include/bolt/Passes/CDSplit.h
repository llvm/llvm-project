//===- bolt/Passes/CDSplit.h - Split functions into hot/warm/cold
// after function reordering pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CDSPLIT
#define BOLT_PASSES_CDSPLIT

#include "bolt/Passes/SplitFunctions.h"
#include <atomic>

namespace llvm {
namespace bolt {

using BasicBlockOrder = BinaryFunction::BasicBlockOrderType;

class CDSplit : public BinaryFunctionPass {
private:
  /// Overall stats.
  std::atomic<uint64_t> SplitBytesHot{0ull};
  std::atomic<uint64_t> SplitBytesCold{0ull};

  /// List of functions to be considered.
  /// All functions in the list are used to construct a call graph.
  /// A subset of functions in this list are considered for splitting.
  std::vector<BinaryFunction *> FunctionsToConsider;

  /// Helper functions to initialize global variables.
  void initialize(BinaryContext &BC);

  /// Split function body into 3 fragments: hot / warm / cold.
  void runOnFunction(BinaryFunction &BF);

  /// Assign each basic block in the given function to either hot, cold,
  /// or warm fragment using the CDSplit algorithm.
  void assignFragmentThreeWay(const BinaryFunction &BF,
                              const BasicBlockOrder &BlockOrder);

  /// Find the best split index that separates hot from warm.
  /// The basic block whose index equals the returned split index will be the
  /// last hot block.
  size_t findSplitIndex(const BinaryFunction &BF,
                        const BasicBlockOrder &BlockOrder);

public:
  explicit CDSplit(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;

  const char *getName() const override { return "cdsplit"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
