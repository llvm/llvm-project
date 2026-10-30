//===---------- Linux implementation of the epoll_ctl function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/epoll/epoll_ctl.h"

#include "hdr/types/struct_epoll_event.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/epoll_ctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, epoll_ctl,
                   (int epfd, int op, int fd, epoll_event *event)) {
  auto result = linux_syscalls::epoll_ctl(epfd, op, fd, event);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
