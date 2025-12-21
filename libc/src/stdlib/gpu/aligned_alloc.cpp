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
  // FIXME: NVIDIA targets currently use the built-in 'malloc' which we cannot
  // reason with. But we still need to provide this function for compatibility.
#ifndef LIBC_TARGET_ARCH_IS_NVPTX
  return gpu::aligned_allocate(static_cast<uint32_t>(alignment), size);
#else
  (void)alignment;
  (void)size;
  return nullptr;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
