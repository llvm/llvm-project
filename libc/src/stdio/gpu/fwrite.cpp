//===-- GPU implementation of fwrite --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fwrite.h"
#include "file.h"

#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(size_t, fwrite,
                   (const void *__restrict buffer, size_t size, size_t nmemb,
                    ::FILE *stream)) {
  if (size == 0 || nmemb == 0)
    return 0;

  auto result = file::write(stream, buffer, size * nmemb);
  return result / size;
}

} // namespace __llvm_libc
