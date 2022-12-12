//===-- Implementation of fgetc -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgetc.h"
#include "src/__support/File/file.h"

#include <errno.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fgetc, (::FILE * stream)) {
  unsigned char c;
  auto result = reinterpret_cast<__llvm_libc::File *>(stream)->read(&c, 1);
  size_t r = result.value;
  if (result.has_error())
    errno = result.error;

  if (r != 1)
    return EOF;
  return c;
}

} // namespace __llvm_libc
