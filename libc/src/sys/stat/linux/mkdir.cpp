//===-- Linux implementation of mkdir -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdir.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, mkdir, (const char *path, mode_t mode)) {
#ifdef SYS_mkdir
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_mkdir, path, mode);
#elif defined(SYS_mkdirat)
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_mkdirat, AT_FDCWD, path, mode);
#else
#error "mkdir and mkdirat syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE
