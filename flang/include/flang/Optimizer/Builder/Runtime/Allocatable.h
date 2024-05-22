//===-- Allocatable.h - generate Allocatable runtime API calls---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate runtime call to assign \p sourceBox to \p destBox.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules, otherwise it is not
/// modified.
mlir::Value genMoveAlloc(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value to, mlir::Value from, mlir::Value hasStat,
                         mlir::Value errMsg);

/// Generate runtime call to apply bounds, cobounds, length type
/// parameters and derived type information from \p mold descriptor
/// to \p desc descriptor. The resulting rank of \p desc descriptor
/// is set to \p rank. The resulting descriptor must be initialized
/// and deallocated before the call.
void genAllocatableApplyMold(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value desc, mlir::Value mold, int rank);

/// Generate runtime call to set the bounds (\p lowerBound and \p upperBound)
/// for the specified dimension \p dimIndex (zero-based) in the given
/// \p desc descriptor.
void genAllocatableSetBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value desc, mlir::Value dimIndex,
                             mlir::Value lowerBound, mlir::Value upperBound);

/// Generate runtime call to allocate an allocatable entity
/// as described by the given \p desc descriptor.
void genAllocatableAllocate(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value desc, mlir::Value hasStat = {},
                            mlir::Value errMsg = {});

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H
