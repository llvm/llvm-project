//===---------- Linux implementation of the epoll_pwait2 function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/epoll/epoll_pwait2.h"

#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_epoll_event.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/epoll_pwait2.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, epoll_pwait2,
                   (int epfd, struct epoll_event *events, int maxevents,
                    const struct timespec *timeout, const sigset_t *sigmask)) {
  auto result =
      linux_syscalls::epoll_pwait2(epfd, events, maxevents, timeout, sigmask);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }

  LIBC_MSAN_UNPOISON(events, result.value() * sizeof(struct epoll_event));

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
