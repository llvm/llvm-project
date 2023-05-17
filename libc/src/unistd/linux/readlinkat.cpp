//===-- Linux implementation of readlinkat --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/readlinkat.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(ssize_t, readlinkat,
                   (int fd, const char *__restrict path, char *__restrict buf,
                    size_t bufsize)) {
  ssize_t ret =
      __llvm_libc::syscall_impl(SYS_readlinkat, fd, path, buf, bufsize);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
