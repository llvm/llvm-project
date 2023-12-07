//===-- Implementation of getchar_unlocked --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getchar_unlocked.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, getchar_unlocked, ()) {
  unsigned char c;
  auto result = stdin->read_unlocked(&c, 1);
  if (result.has_error())
    libc_errno = result.error;

  if (result.value != 1)
    return EOF;
  return c;
}

} // namespace __llvm_libc
