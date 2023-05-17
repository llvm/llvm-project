//===-- include/flang/Runtime/support.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for runtime support code for lowering.
#ifndef FORTRAN_RUNTIME_SUPPORT_H_
#define FORTRAN_RUNTIME_SUPPORT_H_

#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Predicate: is the storage described by a Descriptor contiguous in memory?
bool RTNAME(IsContiguous)(const Descriptor &);

/// Copy elements from \p source to a contiguous memory area denoted
/// by \p destination. The caller must guarantee that the destination
/// buffer is big enough to hold all elements from \p source, and also
/// that its alignment satisfies the minimal alignment required
/// for the elements of \p source. \p destByteSize is the size in bytes
/// of the destination buffer, and is only used for checking for overflows
/// of the buffer.
/// The runtime implementation is optimized to make reads from \p source
/// efficiently by identifying contiguity in the leading dimensions (if any).
///
/// The implementation assumes that \p source and \p destination elements'
/// locations never overlap.
void RTNAME(PackContiguous)(
    void *destination, const Descriptor &source, std::size_t destByteSize);

/// Copy element from contiguous memory area denoted by \p source into
/// \p destination. The caller must guarantee that the source buffer
/// contains enough elements to be copied into \p destination, and also
/// that its alignment satisfies the minimal alignment required
/// for the elements of \p destination. \p destByteSize is the size in bytes
/// of the source buffer, and is only used for checking for overruns
/// of the buffer.
/// The runtime implementation is optimized to make writes into \p destination
/// efficiently by identifying contiguity in the leading dimensions (if any).
///
/// The implementation assumes that \p source and \p destination elements'
/// locations never overlap.
void RTNAME(UnpackContiguous)(const Descriptor &destination, const void *source,
    std::size_t sourceByteSize);

/// If \p source specifies contiguous storage in memory, then
/// the returned address matches source.base_addr, otherwise,
/// if \p destination is not null, then the function copies
/// all elements from \p source into the destination buffer
/// and returns \p destination, otherwise, the function returns
/// a pointer to newly allocated contiguous buffer containing
/// all elements from \p source (e.g. copied with RTNAME(PackContiguous)).
///
/// If \p destination is not null, then the caller must guarantee
/// that the destination buffer is big enough to hold all elements
/// from \p source, and also that its alignment satisfies the minimal
/// alignment required for the elements of \p source.
/// \p destByteSize is the size in bytes of the destination buffer,
/// and is only used for checking for overflows of the buffer.
///
/// \p shouldFree is set to 1, if the function allocates new memory,
/// otherwise, it is set to 0.
void *RTNAME(MakeContiguous)(void *destination, const Descriptor &source,
    std::size_t destByteSize, std::uint8_t *shouldFree);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_SUPPORT_H_
