//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of ppoll.
///
//===----------------------------------------------------------------------===//

#include "src/poll/ppoll.h"

#include "hdr/signal_macros.h"
#include "hdr/types/nfds_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_pollfd.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // SYS_ppoll, SYS_ppoll_time64

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ppoll,
                   (struct pollfd * fds, nfds_t nfds,
                    const struct timespec *tmo_p, const sigset_t *sigmask)) {
  timespec ts;
  timespec *tsp = nullptr;
  if (tmo_p != nullptr) {
    ts = *tmo_p;
    tsp = &ts;
  }

#if defined(SYS_ppoll_time64)
  // The kernel expects the signal mask size in bytes, not the number of
  // signals. NSIG is the signal count, so NSIG / 8 gives the byte size.
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ppoll_time64, fds, nfds, tsp,
                                              sigmask, NSIG / 8);
#elif defined(SYS_ppoll)
  static_assert(
      sizeof(timespec::tv_nsec) == sizeof(long),
      "This legacy syscall fallback is only safe on platforms where tv_nsec "
      "matches the register size (long). It is unsafe on 32-bit platforms "
      "with 64-bit tv_nsec.");
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ppoll, fds, nfds, tsp,
                                              sigmask, NSIG / 8);
#else
#error "ppoll and ppoll_time64 syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
