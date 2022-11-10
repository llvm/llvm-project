//===- AliasAnalysis.h - Alias Analysis in FIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FIR_ANALYSIS_ALIASANALYSIS_H_
#define FIR_ANALYSIS_ALIASANALYSIS_H_

#include "mlir/Analysis/AliasAnalysis.h"

namespace fir {

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//
class AliasAnalysis {
public:
  /// Given two values, return their aliasing behavior.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);
};
} // namespace fir

#endif // FIR_ANALYSIS_ALIASANALYSIS_H_
