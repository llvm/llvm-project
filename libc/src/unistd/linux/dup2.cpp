//===-- Linux implementation of dup2 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/dup2.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, dup2, (int oldfd, int newfd)) {
#ifdef SYS_dup2
  // If dup2 syscall is available, we make use of directly.
  long ret = __llvm_libc::syscall(SYS_dup2, oldfd, newfd);
#elif defined(SYS_dup3)
  // If dup2 syscall is not available, we try using the dup3 syscall. However,
  // dup3 fails if oldfd is the same as newfd. So, we handle that case
  // separately before making the dup3 syscall.
  if (oldfd == newfd) {
    // Check if oldfd is actually a valid file descriptor.
    long ret = __llvm_libc::syscall(SYS_fcntl, oldfd, F_GETFD);
    if (ret >= 0)
      return oldfd;
    errno = -ret;
    return -1;
  }
  long ret = __llvm_libc::syscall(SYS_dup3, oldfd, newfd, 0);
#else
#error "SYS_dup2 and SYS_dup3 not available for the target."
#endif
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
