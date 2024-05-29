//===-- Support.h - generate support runtime API calls ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate call to `CopyAndUpdateDescriptor` runtime routine.
void genCopyAndUpdateDescriptor(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value to, mlir::Value from,
                                mlir::Value newDynamicType,
                                mlir::Value newAttribute,
                                mlir::Value newLowerBounds);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H
