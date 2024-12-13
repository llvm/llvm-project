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
  /// Bit vector of memory addresses of vtables.
  llvm::BitVector VtableBitVector;
  /// Returns true if the memory address is in the bit vector of vtables.
  bool isAddressInVTable(uint64_t Address) const {
    return VtableBitVector.test(Address / 8);
  }
  /// Initialize bit vector of memory addresses of vtables.
  void initVtable() { VtableBitVector.resize((((uint64_t)1) << 32) / 8); }
  /// Mark memory address of vtable as used.
  void setAddressUsedInVTable(uint64_t Address) {
    VtableBitVector.set(Address / 8);
  }
  /// Scans symbol table and creates a bit vector of memory addresses of
  /// vtables.
  void initVTableReferences(const BinaryContext &BC);
  /// Analyze .text section and relocations and mark functions that are not
  /// safe to fold.
  void markFunctionsUnsafeToFold(BinaryContext &BC);
  /// Process static and dynamic relocations in the data sections to identify
  /// function references, and marks them as unsafe to fold. It filters out
  /// symbol references that are in vtables.
  void analyzeDataRelocations(BinaryContext &BC);
  /// Process functions that have CFG created and mark functions unsafe to fold
  /// that are used in non-control flow instructions.
  void analyzeFunctions(BinaryContext &BC);
};

} // namespace bolt
} // namespace llvm

#endif
