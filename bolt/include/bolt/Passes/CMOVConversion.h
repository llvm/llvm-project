//===- bolt/Passes/CMOVConversion.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass finds the following patterns:
//         jcc
//        /   \
// (empty)    mov src, dst
//        \   /
//
// and replaces them with:
//
//   cmovcc src, dst
//
// The advantage of performing this conversion in BOLT (compared to compiler
// heuristic driven instruction selection) is that BOLT can use LBR
// misprediction information and only convert poorly predictable branches.
// Note that branch misprediction rate is different from branch bias.
// For well-predictable branches, it might be beneficial to leave jcc+mov as is
// from microarchitectural perspective to avoid unneeded dependencies (CMOV
// instruction has a dataflow dependence on flags and both operands).
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CMOVCONVERSION_H
#define BOLT_PASSES_CMOVCONVERSION_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Pass for folding eligible hammocks into CMOV's if profitable.
class CMOVConversion : public BinaryFunctionPass {
  struct Stats {
    /// Record how many possible cases there are.
    uint64_t StaticPossible = 0;
    uint64_t DynamicPossible = 0;

    /// Record how many cases were converted.
    uint64_t StaticPerformed = 0;
    uint64_t DynamicPerformed = 0;

    /// Record how many mispredictions were eliminated.
    uint64_t PossibleMP = 0;
    uint64_t RemovedMP = 0;

    Stats operator+(const Stats &O) {
      StaticPossible += O.StaticPossible;
      DynamicPossible += O.DynamicPossible;
      StaticPerformed += O.StaticPerformed;
      DynamicPerformed += O.DynamicPerformed;
      PossibleMP += O.PossibleMP;
      RemovedMP += O.RemovedMP;
      return *this;
    }
    double getStaticRatio() { return (double)StaticPerformed / StaticPossible; }
    double getDynamicRatio() {
      return (double)DynamicPerformed / DynamicPossible;
    }
    double getMPRatio() { return (double)RemovedMP / PossibleMP; }

    void dump();
  };
  // BinaryContext-wide stats
  Stats Global;

  void runOnFunction(BinaryFunction &Function);

public:
  explicit CMOVConversion() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "CMOV conversion"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
