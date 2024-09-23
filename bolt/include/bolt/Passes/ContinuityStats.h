//===- bolt/Passes/ContinuityStats.h - function cfg continuity analysis ---*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conduct function CFG continuity analysis.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CONTINUITYSTATS_H
#define BOLT_PASSES_CONTINUITYSTATS_H

#include "bolt/Passes/BinaryPasses.h"
#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {
class BinaryContext;

/// Compute and report to the user the function CFG continuity quality
class PrintContinuityStats : public BinaryFunctionPass {
public:
  explicit PrintContinuityStats(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "continuity-stats"; }
  bool shouldPrint(const BinaryFunction &) const override { return false; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_CONTINUITYSTATS_H
