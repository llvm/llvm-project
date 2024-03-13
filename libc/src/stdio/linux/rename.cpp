//===-- Linux implementation of rename ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/rename.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <fcntl.h>       // For AT_* macros.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, rename, (const char *oldpath, const char *newpath)) {
#if defined(SYS_rename)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_rename, oldpath, newpath);
#elif defined(SYS_renameat)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_renameat, AT_FDCWD, oldpath,
                                              AT_FDCWD, newpath);
#else
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_renameat2, AT_FDCWD, oldpath,
                                              AT_FDCWD, newpath, 0);
#endif

  if (ret >= 0)
    return 0;
  libc_errno = -ret;
  return -1;
}

} // namespace LIBC_NAMESPACE

