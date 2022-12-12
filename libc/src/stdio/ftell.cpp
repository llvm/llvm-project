//===-- Implementation of ftell -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/ftell.h"
#include "src/__support/File/file.h"

#include <errno.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long, ftell, (::FILE * stream)) {
  auto result = reinterpret_cast<__llvm_libc::File *>(stream)->tell();
  if (!result.has_value()) {
    errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace __llvm_libc
