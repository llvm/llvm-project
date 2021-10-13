//===-- DoLoopHelper.h -- gen fir.do_loop ops -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_DOLOOPHELPER_H
#define FORTRAN_OPTIMIZER_BUILDER_DOLOOPHELPER_H

#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace fir::factory {

/// Helper to build fir.do_loop Ops.
class DoLoopHelper {
public:
  explicit DoLoopHelper(fir::FirOpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}
  DoLoopHelper(const DoLoopHelper &) = delete;

  /// Type of a callback to generate the loop body.
  using BodyGenerator = std::function<void(fir::FirOpBuilder &, mlir::Value)>;

  /// Build loop [\p lb, \p ub] with step \p step.
  /// If \p step is an empty value, 1 is used for the step.
  fir::DoLoopOp createLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
                           const BodyGenerator &bodyGenerator);

  /// Build loop [\p lb,  \p ub] with step 1.
  fir::DoLoopOp createLoop(mlir::Value lb, mlir::Value ub,
                           const BodyGenerator &bodyGenerator);

  /// Build loop [0, \p count) with step 1.
  fir::DoLoopOp createLoop(mlir::Value count,
                           const BodyGenerator &bodyGenerator);

private:
  fir::FirOpBuilder &builder;
  mlir::Location loc;
};

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_DOLOOPHELPER_H
