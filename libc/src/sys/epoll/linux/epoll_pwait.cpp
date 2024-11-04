//===---------- Linux implementation of the epoll_pwait function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/epoll/epoll_pwait.h"

#include "hdr/signal_macros.h" // for NSIG
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_epoll_event.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/sanitizer.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, epoll_pwait,
                   (int epfd, struct epoll_event *events, int maxevents,
                    int timeout, const sigset_t *sigmask)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_epoll_pwait, epfd, reinterpret_cast<long>(events), maxevents, timeout,
      reinterpret_cast<long>(sigmask), NSIG / 8);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  MSAN_UNPOISON(events, ret * sizeof(struct epoll_event));

  return ret;
}

} // namespace LIBC_NAMESPACE
