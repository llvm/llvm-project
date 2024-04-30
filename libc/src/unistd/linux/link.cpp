//===-- Linux implementation of link --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/link.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, link, (const char *path1, const char *path2)) {
#ifdef SYS_link
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_link, path1, path2);
#elif defined(SYS_linkat)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_linkat, AT_FDCWD, path1,
                                              AT_FDCWD, path2, 0);
#else
#error "link or linkat syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
