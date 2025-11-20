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
#include "llvm/ADT/SparseBitVector.h"

namespace llvm {
namespace bolt {

/// An optimization that replaces references to identical functions with
/// references to a single one of them.
///
class IdenticalCodeFolding : public BinaryFunctionPass {
protected:
  /// Return true if the function is safe to fold.
  bool shouldOptimize(const BinaryFunction &BF) const override;

public:
  enum class ICFLevel {
    None, /// No ICF. (Default)
    Safe, /// Safe ICF.
    All,  /// Aggressive ICF.
  };
  explicit IdenticalCodeFolding(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "identical-code-folding"; }
  Error runOnFunctions(BinaryContext &BC) override;

private:
  /// Bit vector of memory addresses of vtables.
  llvm::SparseBitVector<> VTableBitVector;

  /// Return true if the memory address is in a vtable.
  bool isAddressInVTable(uint64_t Address) const {
    return VTableBitVector.test(Address / 8);
  }

  /// Mark memory address of a vtable as used.
  void setAddressUsedInVTable(uint64_t Address) {
    VTableBitVector.set(Address / 8);
  }

  /// Scan symbol table and mark memory addresses of
  /// vtables.
  void initVTableReferences(const BinaryContext &BC);

  /// Analyze code section and relocations and mark functions that are not
  /// safe to fold.
  void markFunctionsUnsafeToFold(BinaryContext &BC);

  /// Process static and dynamic relocations in the data sections to identify
  /// function references, and mark them as unsafe to fold. It filters out
  /// symbol references that are in vtables.
  void analyzeDataRelocations(BinaryContext &BC);

  /// Process functions that have been disassembled and mark functions that are
  /// used in non-control flow instructions as unsafe to fold.
  void analyzeFunctions(BinaryContext &BC);
};

class DeprecatedICFNumericOptionParser
    : public cl::parser<IdenticalCodeFolding::ICFLevel> {
public:
  explicit DeprecatedICFNumericOptionParser(cl::Option &O)
      : cl::parser<IdenticalCodeFolding::ICFLevel>(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg,
             IdenticalCodeFolding::ICFLevel &Value) {
    if (Arg == "0" || Arg == "1") {
      Value = (Arg == "0") ? IdenticalCodeFolding::ICFLevel::None
                           : IdenticalCodeFolding::ICFLevel::All;
      errs() << formatv("BOLT-WARNING: specifying numeric value \"{0}\" "
                        "for option -{1} is deprecated\n",
                        Arg, ArgName);
      return false;
    }
    return cl::parser<IdenticalCodeFolding::ICFLevel>::parse(O, ArgName, Arg,
                                                             Value);
  }
};

} // namespace bolt
} // namespace llvm

#endif
