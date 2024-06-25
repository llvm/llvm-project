//===- bolt/Passes/RegReAssign.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REGREASSIGN_H
#define BOLT_PASSES_REGREASSIGN_H

#include "bolt/Passes/BinaryFunctionCallGraph.h"
#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Passes/RegAnalysis.h"

namespace llvm {
namespace bolt {

class RegReAssign : public BinaryFunctionPass {
  std::vector<int64_t> RegScore;
  std::vector<size_t> RankedRegs;
  BitVector ClassicRegs;
  BitVector CalleeSaved;
  BitVector ClassicCSR;
  BitVector ExtendedCSR;
  BitVector GPRegs;

  /// Hooks to other passes
  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;

  /// Stats
  DenseSet<const BinaryFunction *> FuncsChanged;
  int64_t StaticBytesSaved{0};
  int64_t DynBytesSaved{0};

  void swap(BinaryFunction &Function, MCPhysReg A, MCPhysReg B);
  void rankRegisters(BinaryFunction &Function);
  void aggressivePassOverFunction(BinaryFunction &Function);
  bool conservativePassOverFunction(BinaryFunction &Function);
  void setupAggressivePass(BinaryContext &BC,
                           std::map<uint64_t, BinaryFunction> &BFs);
  void setupConservativePass(BinaryContext &BC,
                             std::map<uint64_t, BinaryFunction> &BFs);

public:
  /// BinaryPass public interface

  explicit RegReAssign(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "regreassign"; }

  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && FuncsChanged.count(&BF) > 0;
  }

  Error runOnFunctions(BinaryContext &BC) override;
};
} // namespace bolt
} // namespace llvm

#endif
