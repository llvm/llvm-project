//===-- Implementation of fgets for baremetal -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fgets.h"

#include "hdr/stdio_macros.h" // for EOF.
#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, fgets,
                   (char *__restrict str, int count,
                    ::FILE *__restrict stream)) {
  if (count < 1)
    return nullptr;

  char c = '\0';
  // i is an int because it's frequently compared to count, which is also int.
  int i = 0;

  for (; i < (count - 1) && c != '\n'; ++i) {
    auto result = read_internal(&c, 1, stream);
    if (result.has_error())
      libc_errno = result.error;

    if (result.value != 1)
      break;
    str[i] = c;
  }

  // If the requested read size makes no sense, an error occured, or no bytes
  // were read due to an EOF, then return nullptr and don't write the null byte.
  if (i == 0)
    return nullptr;

  str[i] = '\0';
  return str;
}

} // namespace LIBC_NAMESPACE_DECL
