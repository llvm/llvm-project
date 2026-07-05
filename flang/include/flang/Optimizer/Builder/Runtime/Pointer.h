//===-- Pointer.h - generate pointer runtime API calls-----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_POINTER_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_POINTER_H

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate runtime call to associate \p target address of scalar
/// with the \p desc pointer descriptor.
void genPointerAssociateScalar(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value desc, mlir::Value target);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_POINTER_H
