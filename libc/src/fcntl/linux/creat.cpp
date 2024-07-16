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
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, creat, (const char *path, int mode_flags)) {
#ifdef SYS_open
  int fd = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_open, path, O_CREAT | O_WRONLY | O_TRUNC, mode_flags);
#else
  int fd = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_openat, AT_FDCWD, path, O_CREAT | O_WRONLY | O_TRUNC, mode_flags);
#endif

  if (fd > 0)
    return fd;

  libc_errno = -fd;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
