//===---------- Linux implementation of the ioctl function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/ioctl/ioctl.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include <stdarg.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ioctl, (int fd, unsigned long request, ...)) {
  va_list vargs;
  va_start(vargs, request);
  void *data_pointer = va_arg(vargs, void *);
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_ioctl, fd, request, data_pointer);
  va_end(vargs);

  // From `man ioctl`:
  // "Usually, on success zero is returned.  A few ioctl() operations
  // use the return value as an output parameter and return a
  // nonnegative value on success.  On error, -1 is returned, and errno
  // is set to indicate the error."
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
