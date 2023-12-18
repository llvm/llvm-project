//===-- GPU implementation of putc ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/putc.h"
#include "file.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, putc, (int c, ::FILE *stream)) {
  unsigned char uc = static_cast<unsigned char>(c);

  size_t written = file::write(stream, &uc, 1);
  if (1 != written)
    return EOF;

  return 0;
}

} // namespace LIBC_NAMESPACE
