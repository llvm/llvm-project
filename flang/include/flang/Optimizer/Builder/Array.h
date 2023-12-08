//===-- Array.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_ARRAY_H
#define FORTRAN_OPTIMIZER_BUILDER_ARRAY_H

#include "flang/Optimizer/Dialect/FIROps.h"

namespace fir::factory {

/// Return true if and only if the extents are those of an assumed-size array.
/// An assumed-size array in Fortran is declared with `*` as the upper bound of
/// the last dimension of the array. Lowering converts the asterisk to an
/// undefined value.
inline bool isAssumedSize(const llvm::SmallVectorImpl<mlir::Value> &extents) {
  return !extents.empty() &&
         mlir::isa_and_nonnull<UndefOp>(extents.back().getDefiningOp());
}

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_ARRAY_H
