//===- CIRBasicAliasAnalysis.h - Basic CIR Alias Analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRBasicAliasAnalysis, a CIR-specific alias analysis
// implementation based on pointer provenance and distinct allocation sites.
// Register with an mlir::AliasAnalysis aggregate via
// addAnalysisImplementation(), or use registerCIRAliasAnalyses() to add the
// full suite of CIR analyses at once.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace cir {

/// Basic CIR alias analysis based on pointer provenance and distinct allocation
/// sites. Conservative defaults (MayAlias / ModRef) are returned for cases
/// that are not yet handled.
class CIRBasicAliasAnalysis {
public:
  explicit CIRBasicAliasAnalysis(mlir::Operation *op)
      : dataLayout(mlir::DataLayout::closest(op)) {}
  CIRBasicAliasAnalysis(CIRBasicAliasAnalysis &&) = default;

  /// Return the aliasing behavior between two values.
  ///
  /// Both values are traced back to the object they point into and to their
  /// byte offset within it. Pointers into provably different objects don't
  /// alias, and pointers at the same offset into the same object must alias.
  /// MayAlias is returned whenever a more precise answer cannot be determined.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  ///
  /// Returns ModRef conservatively. CIR ops that carry explicit memory-effect
  /// attributes or that are known to be pure/read-only can be handled here.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  mlir::DataLayout dataLayout;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H
