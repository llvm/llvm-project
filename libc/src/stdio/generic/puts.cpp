//===-- Implementation of puts --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  cpp::string_view str_view(str);
  auto result = __llvm_libc::stdout->write(str, str_view.size());
  if (result.has_error())
    libc_errno = result.error;
  size_t written = result.value;
  if (str_view.size() != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  result = __llvm_libc::stdout->write("\n", 1);
  if (result.has_error())
    libc_errno = result.error;
  written = result.value;
  if (1 != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  return 0;
}

} // namespace __llvm_libc
