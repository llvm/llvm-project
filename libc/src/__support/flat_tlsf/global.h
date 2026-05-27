//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide Global Heap class for flat_tlsf tracking lazy initialization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_GLOBAL_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_GLOBAL_H

#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/span.h"
#include "src/__support/flat_tlsf/heap.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/math_extras.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" cpp::byte _end;
extern "C" cpp::byte __llvm_libc_heap_limit;

namespace flat_tlsf {

class FlatTlsfHeap {
public:
  LIBC_INLINE constexpr FlatTlsfHeap()
      : begin(&_end), end(&__llvm_libc_heap_limit) {}

  LIBC_INLINE constexpr FlatTlsfHeap(cpp::span<cpp::byte> region)
      : begin(region.data()), end(region.data() + region.size()) {}

  LIBC_INLINE void *allocate(size_t size) {
    if (heap.get_gap_list() == nullptr)
      init();
    return heap.malloc(size);
  }

  LIBC_INLINE void *aligned_allocate(size_t alignment, size_t size) {
    if (size == 0)
      return nullptr;

    // The alignment must be an integral power of two.
    if (!bit_utils::is_power_of_2(alignment))
      return nullptr;

    // The size parameter must be an integral multiple of alignment.
    if (size % alignment != 0)
      return nullptr;

    if (heap.get_gap_list() == nullptr)
      init();
    return heap.aligned_alloc(alignment, size);
  }

  LIBC_INLINE void free(void *ptr) {
    if (ptr != nullptr)
      LIBC_ASSERT(is_valid_ptr(ptr) && "Invalid pointer");
    heap.free(ptr);
  }

  LIBC_INLINE void *realloc(void *ptr, size_t size) {
    if (heap.get_gap_list() == nullptr)
      init();
    if (ptr != nullptr)
      LIBC_ASSERT(is_valid_ptr(ptr) && "Invalid pointer");
    return heap.realloc(ptr, size);
  }

  LIBC_INLINE void *calloc(size_t num, size_t size) {
    size_t bytes;
    if (mul_overflow(num, size, bytes))
      return nullptr;
    void *ptr = allocate(bytes);
    if (ptr != nullptr)
      inline_memset(ptr, 0, bytes);
    return ptr;
  }

  LIBC_INLINE cpp::span<cpp::byte> region() const { return {begin, end}; }

private:
  LIBC_INLINE void init() {
    LIBC_ASSERT(heap.get_gap_list() == nullptr && "duplicate initialization");
    Byte *res = heap.claim(reinterpret_cast<Byte *>(begin),
                           static_cast<size_t>(end - begin));
    LIBC_ASSERT(res != nullptr && "Failed to claim memory for heap");
  }

  LIBC_INLINE bool is_valid_ptr(void *ptr) { return ptr >= begin && ptr < end; }

  cpp::byte *begin;
  cpp::byte *end;
  Heap heap;
};

template <size_t BUFF_SIZE> class FlatTlsfHeapBuffer : public FlatTlsfHeap {
public:
  LIBC_INLINE constexpr FlatTlsfHeapBuffer() : FlatTlsfHeap{buffer}, buffer{} {}

private:
  cpp::byte buffer[BUFF_SIZE];
};

extern FlatTlsfHeap *flat_tlsf_heap;

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_GLOBAL_H
