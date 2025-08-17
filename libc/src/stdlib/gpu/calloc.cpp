//===-- GPU Implementation of calloc --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/calloc.h"

#include "src/__support/GPU/allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, calloc, (size_t num, size_t size)) {
  size_t bytes = num * size;
  if (bytes == 0)
    return nullptr;

  void *ptr = gpu::allocate(bytes);
  if (!ptr)
    return nullptr;

  inline_memset(ptr, 0, bytes);
  return ptr;
}

} // namespace LIBC_NAMESPACE_DECL
