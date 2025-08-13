//===-- Implementation of poll --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/poll/poll.h"

#include "hdr/types/nfds_t.h"
#include "hdr/types/struct_pollfd.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // SYS_poll, SYS_ppoll

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, poll, (pollfd * fds, nfds_t nfds, int timeout)) {
  int ret = 0;

#if defined(SYS_poll)
  ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_poll, fds, nfds, timeout);
#else // no SYS_poll
  timespec ts, *tsp;
  if (timeout >= 0) {
    ts.tv_sec = timeout / 1000;
    ts.tv_nsec = (timeout % 1000) * 1000000;
    tsp = &ts;
  } else {
    tsp = nullptr;
  }
#if defined(SYS_ppoll)
  ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_ppoll, fds, nfds, tsp, nullptr, 0);
#elif defined(SYS_ppoll_time64)
  ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ppoll_time64, fds, nfds, tsp,
                                          nullptr, 0);
#else
#error "poll, ppoll, ppoll_time64 syscalls not available."
#endif // defined(SYS_ppoll) || defined(SYS_ppoll_time64)
#endif // defined(SYS_poll)

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
