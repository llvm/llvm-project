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

/// Generate a runtime call to map an ieee_flag_type exception value to a
/// libm fenv.h value.
mlir::Value genMapException(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value except);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
