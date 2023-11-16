//===-- Implementation of fread_unlocked ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fread_unlocked.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(size_t, fread_unlocked,
                   (void *__restrict buffer, size_t size, size_t nmemb,
                    ::FILE *stream)) {
  if (size == 0 || nmemb == 0)
    return 0;
  auto result = reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->read_unlocked(
      buffer, size * nmemb);
  if (result.has_error())
    libc_errno = result.error;
  return result.value / size;
}

} // namespace LIBC_NAMESPACE
