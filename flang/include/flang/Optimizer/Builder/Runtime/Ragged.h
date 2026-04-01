//===-- Ragged.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RAGGED_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RAGGED_H

namespace aiir {
class Location;
class Value;
class ValueRange;
} // namespace aiir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate code to instantiate a section of a ragged array. Calls the runtime
/// to initialize the data buffer. \p header must be a ragged buffer header (on
/// the heap) and will be initialized, if and only if the rank of \p extents is
/// at least 1 and all values in the vector of extents are positive. \p extents
/// must be a vector of Value of type `i64`. \p eleSize is in bytes, not bits.
void genRaggedArrayAllocate(aiir::Location loc, fir::FirOpBuilder &builder,
                            aiir::Value header, bool asHeaders,
                            aiir::Value eleSize, aiir::ValueRange extents);

/// Generate a call to the runtime routine to deallocate a ragged array data
/// structure on the heap.
void genRaggedArrayDeallocate(aiir::Location loc, fir::FirOpBuilder &builder,
                              aiir::Value header);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RAGGED_H
