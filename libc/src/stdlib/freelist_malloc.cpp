//===-- Implementation for freelist_malloc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "freelist_heap.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/realloc.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {

namespace {
// Users can define LIBC_FREELIST_MALLOC_SIZE for setting the default buffer
// size used by freelist malloc.
#ifdef LIBC_FREELIST_MALLOC_SIZE
constexpr size_t SIZE = LIBC_FREELIST_MALLOC_SIZE;
#else
// TODO: We should probably have something akin to what scudo/sanitizer
// allocators do where each platform defines this.
constexpr size_t SIZE = 0x40000000ULL; // 1GB
#endif
LIBC_CONSTINIT FreeListHeapBuffer<SIZE> freelist_heap_buffer;
} // namespace

FreeListHeap<> *freelist_heap = &freelist_heap_buffer;

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return freelist_heap->allocate(size);
}

LLVM_LIBC_FUNCTION(void, free, (void *ptr)) { return freelist_heap->free(ptr); }

LLVM_LIBC_FUNCTION(void *, calloc, (size_t num, size_t size)) {
  return freelist_heap->calloc(num, size);
}

LLVM_LIBC_FUNCTION(void *, realloc, (void *ptr, size_t size)) {
  return freelist_heap->realloc(ptr, size);
}

} // namespace LIBC_NAMESPACE
