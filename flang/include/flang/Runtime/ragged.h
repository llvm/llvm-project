//===-- Runtime/ragged.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_RAGGED_H_
#define FORTRAN_RUNTIME_RAGGED_H_

#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {

// A ragged array header block.
// The header block is used to create the "array of arrays" ragged data
// structure. It contains a pair in `flags` to indicate if the header points to
// an array of headers (isIndirection) or data elements and the rank of the
// pointed-to array. The rank is the length of the extents vector accessed
// through `extentPointer`. The `bufferPointer` is overloaded
// and is null, points to an array of headers (isIndirection), or data.
// By default, a header is set to zero, which is its unused state.
// The layout of a ragged buffer header is mirrored in the compiler.
struct RaggedArrayHeader {
  std::uint64_t flags;
  void *bufferPointer;
  std::int64_t *extentPointer;
};

extern "C" {

// For more on ragged arrays see https://en.wikipedia.org/wiki/Jagged_array. The
// Flang compiler allocates ragged arrays as a generalization for
// non-rectangular array temporaries. Ragged arrays can be allocated recursively
// and on demand. Structurally, each leaf is an optional rectangular array of
// elements. The shape of each leaf is independent and may be computed on
// demand. Each branch node is an optional, possibly sparse rectangular array of
// headers. The shape of each branch is independent and may be computed on
// demand. Ragged arrays preserve a correspondence between a multidimensional
// iteration space and array access vectors, which is helpful for dependence
// analysis.

// Runtime helper for allocation of ragged array buffers.
// A pointer to the header block to be allocated is given as header. The flag
// isHeader indicates if a block of headers or data is to be allocated. A
// non-negative rank indicates the length of the extentVector, which is a list
// of non-negative extents. elementSize is the size of a data element in the
// rectangular space defined by the extentVector.
void *RTDECL(RaggedArrayAllocate)(void *header, bool isHeader,
    std::int64_t rank, std::int64_t elementSize, std::int64_t *extentVector);

// Runtime helper for deallocation of ragged array buffers. The root header of
// the ragged array structure is passed to deallocate the entire ragged array.
void RTDECL(RaggedArrayDeallocate)(void *raggedArrayHeader);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_RAGGED_H_
