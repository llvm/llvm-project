//===-- Implementation of fwrite_unlocked ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fwrite_unlocked.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(size_t, fwrite_unlocked,
                   (const void *__restrict buffer, size_t size, size_t nmemb,
                    ::FILE *stream)) {

  if (size == 0 || nmemb == 0)
    return 0;
  auto result = reinterpret_cast<__llvm_libc::File *>(stream)->write_unlocked(
      buffer, size * nmemb);
  if (result.has_error())
    libc_errno = result.error;

  return result.value / size;
}

} // namespace __llvm_libc
