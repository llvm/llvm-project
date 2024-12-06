//===- bolt/Passes/IdenticalCodeFolding.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_IDENTICAL_CODE_FOLDING_H
#define BOLT_PASSES_IDENTICAL_CODE_FOLDING_H

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// An optimization that replaces references to identical functions with
/// references to a single one of them.
///
class IdenticalCodeFolding : public BinaryFunctionPass {
protected:
  bool shouldOptimize(const BinaryFunction &BF) const override {
    if (BF.hasUnknownControlFlow())
      return false;
    if (BF.isFolded())
      return false;
    if (BF.hasSDTMarker())
      return false;
    if (BF.hasAddressTaken())
      return false;
    return BinaryFunctionPass::shouldOptimize(BF);
  }

public:
  enum class ICFLevel {
    None, // No ICF. (Default)
    Safe, // Safe ICF for all sections.
    All,  // Aggressive ICF for code.
  };
  explicit IdenticalCodeFolding(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "identical-code-folding"; }
  Error runOnFunctions(BinaryContext &BC) override;

private:
  /// Analyze .text section and relocations and mark functions that are not
  /// safe to fold.
  Error markFunctionsUnsafeToFold(BinaryContext &BC);
  /// Process relocations in the .data section to identify function
  /// references.
  Error processDataRelocations(BinaryContext &BC,
                               const SectionRef &SecRefRelData,
                               const llvm::BitVector &BitVector,
                               const bool HasAddressTaken);
};

} // namespace bolt
} // namespace llvm

#endif
