//===-- GPU Implementation of malloc --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/malloc.h"

#include "src/__support/GPU/allocator.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// FIXME: For now we just default to the NVIDIA device allocator which is
// always available on NVPTX targets. This will be implemented fully later.
#ifndef LIBC_TARGET_ARCH_IS_NVPTX
LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return gpu::allocate(size);
}
#endif

} // namespace LIBC_NAMESPACE_DECL
