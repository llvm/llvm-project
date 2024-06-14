//===-- Implementation for freelist_malloc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "freelist_heap.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {

namespace {
// TODO: We should probably have something akin to what scudo/sanitizer
// allocators do where each platform defines this.
constexpr size_t SIZE = 0x40000000ULL; // 1GB
LIBC_CONSTINIT FreeListHeapBuffer<SIZE> freelist_heap_buffer;
} // namespace

FreeListHeap<> *freelist_heap = &freelist_heap_buffer;

void *malloc(size_t size) { return freelist_heap->allocate(size); }

void free(void *ptr) { freelist_heap->free(ptr); }

void *calloc(size_t num, size_t size) {
  return freelist_heap->calloc(num, size);
}

void *realloc(void *ptr, size_t size) {
  return freelist_heap->realloc(ptr, size);
}

} // namespace LIBC_NAMESPACE
