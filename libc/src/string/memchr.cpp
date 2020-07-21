//===-- Implementation of memchr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memchr.h"
#include "src/__support/common.h"
#include <stddef.h>

namespace __llvm_libc {

// TODO: Look at performance benefits of comparing words.
void *LLVM_LIBC_ENTRYPOINT(memchr)(const void *src, int c, size_t n) {
  const unsigned char *str = reinterpret_cast<const unsigned char *>(src);
  const unsigned char ch = c;
  for (; n && *str != ch; --n, ++str)
    ;
  return n ? const_cast<unsigned char *>(str) : nullptr;
}

} // namespace __llvm_libc
