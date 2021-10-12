//===-- Implementation of memccpy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memccpy.h"

#include "src/__support/common.h"
#include <stddef.h> // For size_t.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, memccpy,
                   (void *__restrict dest, const void *__restrict src, int c,
                    size_t count)) {
  unsigned char end = static_cast<unsigned char>(c);
  const unsigned char *ucSrc = static_cast<const unsigned char *>(src);
  unsigned char *ucDest = static_cast<unsigned char *>(dest);
  size_t i = 0;
  // Copy up until end is found.
  for (; i < count && ucSrc[i] != end; ++i)
    ucDest[i] = ucSrc[i];
  // if i < count, then end must have been found, so copy end into dest and
  // return the byte after.
  if (i < count) {
    ucDest[i] = ucSrc[i];
    return ucDest + i + 1;
  }
  return nullptr;
}

} // namespace __llvm_libc
