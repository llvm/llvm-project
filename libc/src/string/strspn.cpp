//===-- Implementation of strspn ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strspn.h"

#include "src/__support/common.h"
#include "utils/CPP/Bitset.h"
#include <stddef.h>

namespace __llvm_libc {

size_t LLVM_LIBC_ENTRYPOINT(strspn)(const char *src, const char *segment) {
  const char *initial = src;
  cpp::Bitset<256> bitset;

  for (; *segment; ++segment)
    bitset.set(*segment);
  for (; *src && bitset.test(*src); ++src)
    ;
  return src - initial;
}

} // namespace __llvm_libc
