//===-- Lower/ReductionRuntime.h -- lower reduction intrinsics --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_REDUCTIONRUNTIME_H
#define FORTRAN_LOWER_REDUCTIONRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {
  class FirOpBuilder;

  /// Generate call to all runtime routine.
  /// This calls the descriptor based runtime call implementation of the scan
  /// intrinsics.
  void genAllDescriptor(Fortran::lower::FirOpBuilder & builder,
                        mlir::Location loc, mlir::Value resultBox,
                        mlir::Value maskBox, mlir::Value dim);

  /// Generate call to any runtime routine.
  /// This calls the descriptor based runtime call implementation of the scan
  /// intrinsics.
  void genAnyDescriptor(Fortran::lower::FirOpBuilder & builder,
                        mlir::Location loc, mlir::Value resultBox,
                        mlir::Value maskBox, mlir::Value dim);

  /// Generate call to all runtime routine. This version of all is specialized
  /// for rank 1 mask arguments.
  /// This calls the version that returns a scalar logical value.
  mlir::Value genAll(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                     mlir::Value maskBox, mlir::Value dim);

  /// Generate call to any runtime routine. This version of any is specialized
  /// for rank 1 mask arguments.
  /// This calls the version that returns a scalar logical value.
  mlir::Value genAny(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                     mlir::Value maskBox, mlir::Value dim);

  /// Generate call to Maxloc intrinsic runtime routine. This is the version
  /// that does not take a dim argument.
  void genMaxloc(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value arrayBox,
                 mlir::Value maskBox, mlir::Value kind, mlir::Value back);

  /// Generate call to Maxloc intrinsic runtime routine. This is the version
  /// that takes a dim argument.
  void genMaxlocDim(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                    mlir::Value resultBox, mlir::Value arrayBox,
                    mlir::Value dim, mlir::Value maskBox, mlir::Value kind,
                    mlir::Value back);

  /// Generate call to Minloc intrinsic runtime routine. This is the version
  /// that does not take a dim argument.
  void genMinloc(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value arrayBox,
                 mlir::Value maskBox, mlir::Value kind, mlir::Value back);

  /// Generate call to Minloc intrinsic runtime routine. This is the version
  /// that takes a dim argument.
  void genMinlocDim(Fortran::lower::FirOpBuilder & builder, mlir::Location loc,
                    mlir::Value resultBox, mlir::Value arrayBox,
                    mlir::Value dim, mlir::Value maskBox, mlir::Value kind,
                    mlir::Value back);
} // namespace Fortran::lower

#endif // FORTRAN_LOWER_REDUCTIONRUNTIME_H
