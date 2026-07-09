//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A RawMutex protected FreeListHeap backed by sbrk.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_SBRK_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_SBRK_HEAP_H

#include <stddef.h>

#include "src/__support/CPP/mutex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/block.h"
#include "src/__support/freelist_heap.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/os.h"
#include "src/__support/math_extras.h"
#include "src/__support/threads/raw_mutex.h"
#include "src/string/memory_utils/inline_memset.h"

#if defined(LIBC_TARGET_OS_IS_LINUX)
#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include <sys/syscall.h>                        // SYS_brk
#endif

namespace LIBC_NAMESPACE_DECL {

class SbrkHeap {
public:
  LIBC_INLINE constexpr SbrkHeap(size_t initial_size = 4096)
      : break_ptr(nullptr), current_heap_size(initial_size), heap_storage{},
        mtx() {}

  LIBC_INLINE void *allocate(size_t size) {
    cpp::lock_guard lock(mtx);
    if (LIBC_UNLIKELY(break_ptr == nullptr))
      if (!grow_heap())
        return nullptr;
    void *ptr = heap()->allocate(size);
    while (ptr == nullptr) {
      if (!grow_heap())
        return nullptr;
      ptr = heap()->allocate(size);
    }
    return ptr;
  }

  LIBC_INLINE void *aligned_allocate(size_t alignment, size_t size) {
    cpp::lock_guard lock(mtx);
    if (LIBC_UNLIKELY(break_ptr == nullptr)) {
      if (!grow_heap())
        return nullptr;
    }
    void *ptr = heap()->aligned_allocate(alignment, size);
    while (ptr == nullptr) {
      if (!grow_heap())
        return nullptr;
      ptr = heap()->aligned_allocate(alignment, size);
    }
    return ptr;
  }

  LIBC_INLINE void free(void *ptr) {
    cpp::lock_guard lock(mtx);
    if (LIBC_LIKELY(break_ptr != nullptr))
      heap()->free(ptr);
  }

  LIBC_INLINE void *realloc(void *ptr, size_t size) {
    cpp::lock_guard lock(mtx);
    if (LIBC_UNLIKELY(break_ptr == nullptr))
      if (!grow_heap())
        return nullptr;

    void *new_ptr = heap()->realloc(ptr, size);
    while (new_ptr == nullptr && size != 0) {
      if (!grow_heap())
        return nullptr;
      new_ptr = heap()->realloc(ptr, size);
    }
    return new_ptr;
  }

  LIBC_INLINE void *calloc(size_t num, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(num, size, &bytes))
      return nullptr;
    void *ptr = allocate(bytes);
    if (ptr != nullptr)
      LIBC_NAMESPACE::inline_memset(ptr, 0, bytes);
    return ptr;
  }

  LIBC_INLINE span<cpp::byte> region() const {
    if (break_ptr == nullptr)
      return {};
    return heap()->region();
  }

private:
  LIBC_INLINE FreeListHeap *heap() const {
    return reinterpret_cast<FreeListHeap *>(
        const_cast<cpp::byte *>(heap_storage));
  }

  LIBC_INLINE cpp::byte *sbrk(ptrdiff_t increment) {
#if defined(SYS_brk) && defined(LIBC_TARGET_OS_IS_LINUX)
    long curr_brk = syscall_impl<long>(SYS_brk, 0);
    if (curr_brk < 0)
      return nullptr;
    if (increment == 0)
      return reinterpret_cast<cpp::byte *>(curr_brk);
    long new_brk = syscall_impl<long>(SYS_brk, curr_brk + increment);
    if (new_brk != curr_brk + increment)
      return nullptr;
    return reinterpret_cast<cpp::byte *>(curr_brk);
#else
    static constexpr size_t VIRTUAL_HEAP_SIZE = 131072;
    alignas(FreeListHeap) static cpp::byte virtual_heap[VIRTUAL_HEAP_SIZE];
    static cpp::byte *virtual_brk = virtual_heap;
    if (static_cast<size_t>(increment) >
        VIRTUAL_HEAP_SIZE - static_cast<size_t>(virtual_brk - virtual_heap))
      return nullptr;
    cpp::byte *old_brk = virtual_brk;
    virtual_brk += increment;
    return old_brk;
#endif
  }

  LIBC_INLINE bool grow_heap() {
    bool first_time = (break_ptr == nullptr);
    if (first_time) {
      break_ptr = sbrk(0);
      if (break_ptr == nullptr)
        return false;
    }

    size_t increment = current_heap_size;
    cpp::byte *new_brk = sbrk(increment);
    if (new_brk == nullptr || new_brk != break_ptr)
      return false;

    span<cpp::byte> new_mem(break_ptr, increment);
    if (first_time)
      new (heap_storage) FreeListHeap(new_mem);
    else if (!heap()->adopt(new_mem))
      return false;

    break_ptr += increment;
    if (break_ptr != nullptr)
      current_heap_size += increment;
    return true;
  }

  cpp::byte *break_ptr;
  size_t current_heap_size;
  alignas(FreeListHeap) cpp::byte heap_storage[sizeof(FreeListHeap)];
  RawMutex mtx;
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_SBRK_HEAP_H
