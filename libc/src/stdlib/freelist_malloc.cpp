//===-- Implementation for freelist_malloc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/aligned_alloc.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/realloc.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

static LIBC_CONSTINIT FreeListHeap<> freelist_heap_symbols;
FreeListHeap<> *freelist_heap = &freelist_heap_symbols;

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

LLVM_LIBC_FUNCTION(void *, aligned_alloc, (size_t alignment, size_t size)) {
  return freelist_heap->aligned_allocate(alignment, size);
}

} // namespace LIBC_NAMESPACE_DECL
