//===-- Implementation of strspn ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strspn.h"

#include "src/__support/CPP/bitset.h"
#include "src/__support/common.h"
#include <stddef.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(size_t, strspn, (const char *src, const char *segment)) {
  const char *initial = src;
  cpp::bitset<256> bitset;

  for (; *segment; ++segment)
    bitset.set(*reinterpret_cast<const unsigned char *>(segment));
  for (; *src && bitset.test(*reinterpret_cast<const unsigned char *>(src));
       ++src)
    ;
  return src - initial;
}

} // namespace __llvm_libc
