//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for ppoll.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PPOLL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PPOLL_H

#include "hdr/signal_macros.h"
#include "hdr/types/nfds_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_pollfd.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> ppoll(struct pollfd *fds, nfds_t nfds,
                               const struct timespec *__restrict tmo_p,
                               const sigset_t *__restrict sigmask) {
#if defined(SYS_ppoll_time64)
  static_assert(
      sizeof(time_t) == sizeof(int64_t),
      "SYS_ppoll_time64 requires struct timespec with 64-bit members.");
  // The kernel expects the signal mask size in bytes, not the number of
  // signals. NSIG is the signal count, so NSIG / 8 gives the byte size.
  return syscall_checked<int>(SYS_ppoll_time64, fds, nfds, tmo_p, sigmask,
                              NSIG / 8);
#elif defined(SYS_ppoll)
  static_assert(
      sizeof(timespec::tv_nsec) == sizeof(long),
      "This legacy syscall fallback is only safe on platforms where tv_nsec "
      "matches the register size (long). It is unsafe on 32-bit platforms "
      "with 64-bit tv_nsec.");
  // The kernel expects the signal mask size in bytes, not the number of
  // signals. NSIG is the signal count, so NSIG / 8 gives the byte size.
  return syscall_checked<int>(SYS_ppoll, fds, nfds, tmo_p, sigmask, NSIG / 8);
#else
#error "ppoll and ppoll_time64 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PPOLL_H
