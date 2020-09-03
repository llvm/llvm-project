//===-- Lower/DoLoopHelper.h -- gen fir.do_loop ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DOLOOPHELPER_H
#define FORTRAN_LOWER_DOLOOPHELPER_H

#include "flang/Lower/FIRBuilder.h"

namespace Fortran::lower {

/// Helper to build fir.do_loop Ops.
class DoLoopHelper {
public:
  explicit DoLoopHelper(FirOpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}
  DoLoopHelper(const DoLoopHelper &) = delete;

  /// Type of a callback to generate the loop body.
  using BodyGenerator = std::function<void(FirOpBuilder &, mlir::Value)>;

  /// Build loop [\p lb, \p ub] with step \p step.
  /// If \p step is an empty value, 1 is used for the step.
  void createLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [\p lb,  \p ub] with step 1.
  void createLoop(mlir::Value lb, mlir::Value ub,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [0, \p count) with step 1.
  void createLoop(mlir::Value count, const BodyGenerator &bodyGenerator);

private:
  FirOpBuilder &builder;
  mlir::Location loc;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_DOLOOPHELPER_H
