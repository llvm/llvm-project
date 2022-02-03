//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate call to EXIT intrinsic runtime routine.
void genExit(fir::FirOpBuilder &, mlir::Location, mlir::Value status);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H
