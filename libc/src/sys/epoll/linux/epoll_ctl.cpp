//===---------- Linux implementation of the epoll_ctl function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/epoll/epoll_ctl.h"

#include "include/llvm-libc-types/struct_epoll_event.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, epoll_ctl,
                   (int epfd, int op, int fd, epoll_event *event)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_epoll_ctl, epfd, op, fd,
                                              reinterpret_cast<long>(event));

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return ret;
}

} // namespace LIBC_NAMESPACE
