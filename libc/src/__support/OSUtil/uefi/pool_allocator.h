//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Allocator built on the UEFI boot services pool.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_POOL_ALLOCATOR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_POOL_ALLOCATOR_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace uefi {

// malloc() and friends are implemented on top of the firmware's
// AllocatePool()/FreePool() boot services. Every allocation is prefixed by a
// small header that records the pool buffer it came from and the requested
// size, which is what makes realloc(), aligned_alloc() and
// malloc_usable_size() possible on top of a plain pool.
//
// All functions return nullptr on failure. Pointers passed to pool_free(),
// pool_realloc() and pool_allocation_size() must have been obtained from
// pool_allocate() or pool_realloc().

// Allocates `size` bytes aligned to `alignment`, which must be a power of two.
// Alignments smaller than the fundamental alignment are rounded up to it.
void *pool_allocate(size_t alignment, size_t size);

// Allocates `size` bytes with the fundamental alignment.
void *pool_allocate(size_t size);

// Allocates zero-initialized memory for `num` objects of `size` bytes.
void *pool_calloc(size_t num, size_t size);

// Resizes an allocation, preserving its contents up to the smaller of the old
// and new sizes. A null `ptr` behaves like pool_allocate(size) and a zero
// `size` frees the allocation and returns nullptr.
void *pool_realloc(void *ptr, size_t size);

// Releases an allocation. Passing nullptr is a no-op.
void pool_free(void *ptr);

// Returns the size that was requested for an allocation.
size_t pool_allocation_size(const void *ptr);

} // namespace uefi
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_POOL_ALLOCATOR_H
