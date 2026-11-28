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

#include "pool_allocator.h"

#include "config/app.h"
#include "hdr/stdint_proxy.h"
#include "include/llvm-libc-types/EFI_MEMORY_TYPE.h"
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "include/llvm-libc-types/EFI_SYSTEM_TABLE.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {
namespace uefi {
namespace {

// Bookkeeping stored immediately before every pointer handed out.
struct AllocationHeader {
  // The buffer returned by AllocatePool() that holds this allocation.
  void *base;
  // The size requested by the caller.
  size_t size;
};

// AllocatePool() only guarantees 8 byte alignment, so the header is used to
// re-align the user pointer. The smallest alignment handed out is the
// fundamental alignment, which also keeps the header itself aligned.
constexpr size_t MIN_ALIGNMENT =
    cpp::max(alignof(max_align_t), alignof(AllocationHeader));

static_assert(sizeof(AllocationHeader) % alignof(AllocationHeader) == 0,
              "The header must be a multiple of its alignment.");

// Memory type used for the heap. Loader data stays usable by a boot loader
// that keeps running after ExitBootServices(), which is what EFI applications
// are loaded as themselves.
constexpr EFI_MEMORY_TYPE POOL_TYPE = EfiLoaderData;

LIBC_INLINE AllocationHeader *header_of(void *ptr) {
  return static_cast<AllocationHeader *>(ptr) - 1;
}

LIBC_INLINE const AllocationHeader *header_of(const void *ptr) {
  return static_cast<const AllocationHeader *>(ptr) - 1;
}

} // namespace

void *pool_allocate(size_t alignment, size_t size) {
  if (!cpp::has_single_bit(alignment))
    return nullptr;
  alignment = cpp::max(alignment, MIN_ALIGNMENT);

  // Reserve room for the header plus the worst case re-alignment slack.
  size_t total = 0;
  if (__builtin_add_overflow(size, sizeof(AllocationHeader), &total) ||
      __builtin_add_overflow(total, alignment - 1, &total))
    return nullptr;

  void *base = nullptr;
  EFI_STATUS status =
      app.system_table->BootServices->AllocatePool(POOL_TYPE, total, &base);
  if (status != EFI_SUCCESS || base == nullptr)
    return nullptr;

  cpp::byte *unaligned =
      static_cast<cpp::byte *>(base) + sizeof(AllocationHeader);
  size_t offset = (-cpp::bit_cast<uintptr_t>(unaligned)) & (alignment - 1);
  void *ptr = unaligned + offset;
  *header_of(ptr) = {base, size};
  return ptr;
}

void *pool_allocate(size_t size) { return pool_allocate(MIN_ALIGNMENT, size); }

void *pool_calloc(size_t num, size_t size) {
  size_t total = 0;
  if (__builtin_mul_overflow(num, size, &total))
    return nullptr;
  void *ptr = pool_allocate(total);
  if (ptr != nullptr)
    inline_memset(ptr, 0, total);
  return ptr;
}

void *pool_realloc(void *ptr, size_t size) {
  if (ptr == nullptr)
    return pool_allocate(size);
  if (size == 0) {
    pool_free(ptr);
    return nullptr;
  }

  size_t old_size = header_of(ptr)->size;
  if (size <= old_size) {
    // Shrinking in place is always possible: just remember the new size so
    // that malloc_usable_size() keeps reporting what the caller asked for.
    header_of(ptr)->size = size;
    return ptr;
  }

  void *new_ptr = pool_allocate(size);
  if (new_ptr == nullptr)
    return nullptr;
  inline_memcpy(new_ptr, ptr, old_size);
  pool_free(ptr);
  return new_ptr;
}

void pool_free(void *ptr) {
  if (ptr == nullptr)
    return;
  app.system_table->BootServices->FreePool(header_of(ptr)->base);
}

size_t pool_allocation_size(const void *ptr) {
  if (ptr == nullptr)
    return 0;
  return header_of(ptr)->size;
}

} // namespace uefi
} // namespace LIBC_NAMESPACE_DECL
