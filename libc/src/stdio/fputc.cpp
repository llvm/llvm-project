//===-- Implementation of fputc -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fputc.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fputc, (int c, ::FILE *stream)) {
  unsigned char uc = static_cast<unsigned char>(c);

  auto result = reinterpret_cast<__llvm_libc::File *>(stream)->write(&uc, 1);
  if (result.has_error())
    libc_errno = result.error;
  size_t written = result.value;

  if (1 != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  return 0;
}

} // namespace __llvm_libc
