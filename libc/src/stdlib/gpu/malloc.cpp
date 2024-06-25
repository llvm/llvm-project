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

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return gpu::allocate(size);
}

} // namespace LIBC_NAMESPACE
