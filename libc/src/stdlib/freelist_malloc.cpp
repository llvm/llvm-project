//===-- Implementation for freelist_malloc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/realloc.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {

namespace {
#ifdef LIBC_FREELIST_MALLOC_SIZE
// This is set via the LIBC_CONF_FREELIST_MALLOC_BUFFER_SIZE configuration.
constexpr size_t SIZE = LIBC_FREELIST_MALLOC_SIZE;
#else
#error "LIBC_FREELIST_MALLOC_SIZE was not defined for this build."
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
