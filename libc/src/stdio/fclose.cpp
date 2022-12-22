//===-- Implementation of fclose ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/__support/File/file.h"

#include <errno.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fclose, (::FILE * stream)) {
  auto *file = reinterpret_cast<__llvm_libc::File *>(stream);
  int result = File::cleanup(file);
  if (result != 0) {
    errno = result;
    return EOF;
  }
  return 0;
}

} // namespace __llvm_libc
