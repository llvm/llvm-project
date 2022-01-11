//===- bolt/Passes/Aligner.h - Pass for optimal code alignment --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Aligner class, which provides
// alignment for code, e.g. basic block and functions, with the goal to achieve
// the optimal performance.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_ALIGNER_H
#define BOLT_PASSES_ALIGNER_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class AlignerPass : public BinaryFunctionPass {
private:
  /// Stats for usage of max bytes for basic block alignment.
  std::vector<uint32_t> AlignHistogram;
  std::shared_timed_mutex AlignHistogramMtx;

  /// Stats: execution count of blocks that were aligned.
  std::atomic<uint64_t> AlignedBlocksCount{0};

  /// Assign alignment to basic blocks based on profile.
  void alignBlocks(BinaryFunction &Function, const MCCodeEmitter *Emitter);

public:
  explicit AlignerPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "aligner"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
