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
  // FIXME: NVIDIA targets currently use the built-in 'malloc' which we cannot
  // reason with. But we still need to provide this function for compatibility.
#ifndef LIBC_TARGET_ARCH_IS_NVPTX
  return gpu::reallocate(ptr, size);
#else
  (void)ptr;
  (void)size;
  return nullptr;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
