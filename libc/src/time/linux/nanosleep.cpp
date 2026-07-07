//===-- Linux implementation of nanosleep function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/nanosleep.h"
#include "hdr/stdint_proxy.h" // For int64_t.
#include "hdr/time_macros.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

#include "src/__support/time/linux/kernel_timespec.h"
#include <sys/syscall.h>
#if defined(SYS_clock_nanosleep_time64)
#include <linux/time_types.h>
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, nanosleep, (const timespec *req, timespec *rem)) {
  LIBC_CRASH_ON_NULLPTR(req);
#if defined(SYS_clock_nanosleep_time64)
  int ret;
  // In overlay mode on 32-bit platforms, the system timespec (which we use)
  // may be 32-bit (8 bytes), while the kernel's __kernel_timespec (used by
  // _time64 syscalls) is 64-bit (16 bytes). If they match, we can pass the
  // pointer directly. Otherwise, we must convert to avoid stack corruption.
  if constexpr (sizeof(timespec) == sizeof(__kernel_timespec)) {
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_clock_nanosleep_time64,
                                            CLOCK_REALTIME, 0, req, rem);
  } else {
    __kernel_timespec ts64_req{};
    __kernel_timespec *req_ptr = nullptr;
    if (req != nullptr) {
      ts64_req = to_kernel_timespec(*req);
      req_ptr = &ts64_req;
    }
    __kernel_timespec ts64_rem{};
    __kernel_timespec *rem_ptr = rem ? &ts64_rem : nullptr;
    ret = LIBC_NAMESPACE::syscall_impl<int>(
        SYS_clock_nanosleep_time64, CLOCK_REALTIME, 0, req_ptr, rem_ptr);
    if (ret == -EINTR && rem != nullptr) {
      *rem = to_timespec(ts64_rem);
    }
  }
#elif defined(SYS_nanosleep)
  static_assert(
      sizeof(timespec::tv_nsec) == sizeof(long),
      "This legacy syscall fallback is only safe on platforms where tv_nsec "
      "matches the register size (long). It is unsafe on 32-bit platforms "
      "with 64-bit tv_nsec.");
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_nanosleep, req, rem);
#else
#error "SYS_nanosleep and SYS_clock_nanosleep_time64 syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
