//===-- Implementation for freelist_heap ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

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

} // namespace LIBC_NAMESPACE_DECL
