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
#include "src/__support/libc_errno.h"
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

  // Some ioctls can be expected to return positive values
  if (ret >= 0)
    return ret;

  // If there is an error, errno is set and -1 is returned.
  libc_errno = -ret;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
