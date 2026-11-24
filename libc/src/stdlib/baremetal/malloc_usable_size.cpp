//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation for freelist_malloc_usable_size.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/malloc_usable_size.h"
#include "src/__support/freelist_heap.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, malloc_usable_size, (void *ptr)) {
  return freelist_heap->allocation_size(ptr);
}

} // namespace LIBC_NAMESPACE_DECL
