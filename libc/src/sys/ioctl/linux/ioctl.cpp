//===---------- Linux implementation of the POSIX ioctl function --------===//
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

namespace LIBC_NAMESPACE {

// This function is currently linux only. It has to be refactored suitably if
// ioctl is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(int, ioctl, (int fd, unsigned long request, ...)) {
  va_list ptr_to_memory;
  va_start(ptr_to_memory, request);
  va_arg(ptr_to_memory, void *);
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_ioctl, fd, request, ptr_to_memory);
  va_end(ptr_to_memory);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE
