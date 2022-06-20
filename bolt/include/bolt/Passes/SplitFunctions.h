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

/// Split function code in multiple parts.
class SplitFunctions : public BinaryFunctionPass {
public:
  /// Settings for splitting function bodies into hot/cold partitions.
  enum SplittingType : char {
    ST_NONE = 0, /// Do not split functions.
    ST_LARGE,    /// In non-relocation mode, only split functions that
                 /// are too large to fit into the original space.
    ST_ALL,      /// Split all functions.
  };

private:
  /// Split function body into fragments.
  void splitFunction(BinaryFunction &Function);

  /// Create trampoline landing pads for exception handling code to guarantee
  /// that every landing pad is placed in the same function fragment as the
  /// corresponding thrower block. The trampoline landing pad, when created,
  /// will redirect the execution to the real landing pad in a different
  /// fragment.
  void createEHTrampolines(BinaryFunction &Function) const;

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
