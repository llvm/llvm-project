//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memmove.h"
#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dst, const void *src, size_t count)) {
  if (is_disjoint(dst, src, count))
    inline_memcpy(dst, src, count);
  else
    inline_memmove(dst, src, count);
  return dst;
}

} // namespace LIBC_NAMESPACE
