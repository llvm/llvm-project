//===- bolt/Passes/SplitFunctions.h - Split function code -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_SPLIT_FUNCTIONS_H
#define BOLT_PASSES_SPLIT_FUNCTIONS_H

#include "bolt/Passes/BinaryPasses.h"
#include "llvm/Support/CommandLine.h"
#include <atomic>

namespace llvm {
namespace bolt {

/// Strategy used to partition blocks into fragments.
enum SplitFunctionsStrategy : char {
  /// Split each function into a hot and cold fragment using profiling
  /// information.
  Profile2 = 0,
  /// Split each function into a hot and cold fragment at a randomly chosen
  /// split point (ignoring any available profiling information).
  Random2,
  /// Split each function into N fragments at a randomly chosen split points
  /// (ignoring any available profiling information).
  RandomN,
  /// Split all basic blocks of each function into fragments such that each
  /// fragment contains exactly a single basic block.
  All
};

/// Split function code in multiple parts.
class SplitFunctions : public BinaryFunctionPass {
private:
  /// Split function body into fragments.
  template <typename Strategy>
  void splitFunction(BinaryFunction &Function, Strategy S = {});

  /// Map basic block labels to their trampoline block labels.
  using TrampolineSetType = DenseMap<const MCSymbol *, const MCSymbol *>;

  using BasicBlockOrderType = BinaryFunction::BasicBlockOrderType;

  /// Create trampoline landing pads for exception handling code to guarantee
  /// that every landing pad is placed in the same function fragment as the
  /// corresponding thrower block. The trampoline landing pad, when created,
  /// will redirect the execution to the real landing pad in a different
  /// fragment.
  TrampolineSetType createEHTrampolines(BinaryFunction &Function) const;

  /// Merge trampolines into \p Layout without trampolines. The merge will place
  /// a trampoline immediately before its destination. Used to revert the effect
  /// of trampolines after createEHTrampolines().
  BasicBlockOrderType
  mergeEHTrampolines(BinaryFunction &BF, BasicBlockOrderType &Layout,
                     const TrampolineSetType &Trampolines) const;

  std::atomic<uint64_t> SplitBytesHot{0ull};
  std::atomic<uint64_t> SplitBytesCold{0ull};

public:
  explicit SplitFunctions(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;

  const char *getName() const override { return "split-functions"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
