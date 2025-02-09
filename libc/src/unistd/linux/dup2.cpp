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

#include "hdr/fcntl_macros.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dup2, (int oldfd, int newfd)) {
#ifdef SYS_dup2
  // If dup2 syscall is available, we make use of directly.
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_dup2, oldfd, newfd);
#elif defined(SYS_dup3)
  // If dup2 syscall is not available, we try using the dup3 syscall. However,
  // dup3 fails if oldfd is the same as newfd. So, we handle that case
  // separately before making the dup3 syscall.
  if (oldfd == newfd) {
    // Check if oldfd is actually a valid file descriptor.
#if SYS_fcntl
    int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl, oldfd, F_GETFD);
#elif defined(SYS_fcntl64)
    // Same as fcntl but can handle large offsets
    int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fcntl64, oldfd, F_GETFD);
#else
#error "SYS_fcntl and SYS_fcntl64 syscalls not available."
#endif
    if (ret >= 0)
      return oldfd;
    libc_errno = -ret;
    return -1;
  }
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_dup3, oldfd, newfd, 0);
#else
#error "dup2 and dup3 syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
