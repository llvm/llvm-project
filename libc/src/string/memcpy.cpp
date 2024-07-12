//===-- Implementation of memcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcpy.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, memcpy,
                   (void *__restrict dst, const void *__restrict src,
                    size_t size)) {
  inline_memcpy(dst, src, size);
  return dst;
}

} // namespace LIBC_NAMESPACE
