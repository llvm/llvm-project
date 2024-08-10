//===-- GPU Implementation of aligned_alloc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/aligned_alloc.h"

#include "src/__support/GPU/allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, aligned_alloc, (size_t alignment, size_t size)) {
  if ((alignment & -alignment) != alignment)
    return nullptr;

  void *ptr = gpu::allocate(size);
  if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
    gpu::deallocate(ptr);
    return nullptr;
  }
  return ptr;
}

} // namespace LIBC_NAMESPACE_DECL
