//===-- Exceptions.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate a runtime call to map a set of ieee_flag_type exceptions to a
/// libm fenv.h excepts value.
mlir::Value genMapExcept(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value excepts);

mlir::Value genSupportHalting(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value excepts);

mlir::Value genGetUnderflowMode(fir::FirOpBuilder &builder, mlir::Location loc);
void genSetUnderflowMode(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value bit);

mlir::Value genGetModesTypeSize(fir::FirOpBuilder &builder, mlir::Location loc);
mlir::Value genGetStatusTypeSize(fir::FirOpBuilder &builder,
                                 mlir::Location loc);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
