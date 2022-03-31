//===-- Implementation of creat -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/creat.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, creat, (const char *path, int mode_flags)) {
#ifdef SYS_open
  int fd = __llvm_libc::syscall(SYS_open, path, O_CREAT | O_WRONLY | O_TRUNC,
                                mode_flags);
#else
  int fd = __llvm_libc::syscall(SYS_openat, AT_FDCWD, path,
                                O_CREAT | O_WRONLY | O_TRUNC, mode_flags);
#endif

  if (fd > 0)
    return fd;

  errno = -fd;
  return -1;
}

} // namespace __llvm_libc
