//===-- Implementation of fread_unlocked ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fread_unlocked.h"
#include "src/__support/File/file.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(size_t, fread_unlocked,
                   (void *__restrict buffer, size_t size, size_t nmemb,
                    ::FILE *stream)) {
  return reinterpret_cast<__llvm_libc::File *>(stream)->read_unlocked(
      buffer, size * nmemb);
}

} // namespace __llvm_libc
