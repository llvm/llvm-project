//===---------- Linux implementation of the epoll_create function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/epoll/epoll_create.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, epoll_create, ([[maybe_unused]] int size)) {
#ifdef SYS_epoll_create
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_epoll_create, size);
#elif defined(SYS_epoll_create1)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_epoll_create1, 0);
#else
#error                                                                         \
    "epoll_create and epoll_create1 are unavailable. Unable to build epoll_create."
#endif

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
