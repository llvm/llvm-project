//===-- GPU implementation of getc ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/getc.h"
#include "file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, getc, (::FILE * stream)) {
  unsigned char c;
  size_t r = file::read(stream, &c, 1);

  if (r != 1)
    return EOF;
  return c;
}

} // namespace LIBC_NAMESPACE
