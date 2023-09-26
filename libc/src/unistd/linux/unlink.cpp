//===-- Linux implementation of unlink ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/unlink.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, unlink, (const char *path)) {
#ifdef SYS_unlink
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_unlink, path);
#elif defined(SYS_unlinkat)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_unlinkat, AT_FDCWD, path, 0);
#else
#error "unlink and unlinkat syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE
