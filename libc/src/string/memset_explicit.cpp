//===-- Implementation of memset_explicit ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memset_explicit.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/flush_cache.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, memset_explicit,
                   (void *dst, int value, size_t count)) {
  // Use the inline memset function to set the memory.
  inline_memset<true>(dst, static_cast<uint8_t>(value), count);

  // Flush the cache line.
  flush_cache(dst, count);

  return dst;
}

} // namespace LIBC_NAMESPACE
