//===-- Implementation of setvbuf -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/setvbuf.h"
#include "src/__support/File/file.h"

#include <errno.h>
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, setvbuf,
                   (::FILE *__restrict stream, char *__restrict buf, int type,
                    size_t size)) {
  int err = reinterpret_cast<__llvm_libc::File *>(stream)->set_buffer(buf, size,
                                                                      type);
  if (err != 0)
    errno = err;
  return err;
}

} // namespace __llvm_libc
