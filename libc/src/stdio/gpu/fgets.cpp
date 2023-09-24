//===-- GPU implementation of fgets ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgets.h"
#include "file.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"

#include <stddef.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, fgets,
                   (char *__restrict str, int count,
                    ::FILE *__restrict stream)) {
  if (count < 1)
    return nullptr;

  // This implementation is very slow as it makes multiple RPC calls.
  unsigned char c = '\0';
  int i = 0;
  for (; i < count - 1 && c != '\n'; ++i) {
    auto r = file::read(stream, &c, 1);
    if (r != 1)
      break;

    str[i] = c;
  }

  bool has_error = __llvm_libc::ferror(stream);
  bool has_eof = __llvm_libc::feof(stream);

  if (has_error || (i == 0 && has_eof))
    return nullptr;

  str[i] = '\0';
  return str;
}

} // namespace __llvm_libc
