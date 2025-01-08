//===-- GPU Implementation of realloc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/realloc.h"

#include "src/__support/GPU/allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, realloc, (void *ptr, size_t size)) {
  if (ptr == nullptr)
    return gpu::allocate(size);

  void *newmem = gpu::allocate(size);
  if (newmem == nullptr)
    return nullptr;

  // This will copy garbage if it goes beyond the old allocation size.
  inline_memcpy(newmem, ptr, size);
  gpu::deallocate(ptr);
  return newmem;
}

} // namespace LIBC_NAMESPACE_DECL
