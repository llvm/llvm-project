//===-- Linux implementation of sendfile ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/sendfile/sendfile.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/sendfile.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(ssize_t, sendfile,
                   (int out_fd, int in_fd, off_t *offset, size_t count)) {
#ifdef SYS_sendfile
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_sendfile, in_fd,
                                                      out_fd, offset, count);
#elif defined(SYS_sendfile64)
  // Same as sendfile but can handle large offsets
  static_assert(sizeof(off_t) == 8);
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_sendfile64, in_fd,
                                                      out_fd, offset, count);
#else
#error "sendfile and sendfile64 syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
