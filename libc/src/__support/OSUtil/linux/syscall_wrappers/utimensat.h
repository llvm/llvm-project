//===-- Implementation header for utimensat ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_UTIMENSAT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_UTIMENSAT_H

#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

#include "hdr/types/struct_timespec.h"
#include "src/__support/time/linux/kernel_timespec.h"
#include <sys/syscall.h>
#if defined(SYS_utimensat_time64)
#include <linux/time_types.h>
#endif

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> utimensat(int dirfd, const char *path,
                                   const timespec times[2], int flags) {
#if defined(SYS_utimensat_time64)
  int ret;
  // In overlay mode on 32-bit platforms, the system timespec (which we use)
  // may be 32-bit (8 bytes), while the kernel's __kernel_timespec (used by
  // _time64 syscalls) is 64-bit (16 bytes). If they match, we can pass the
  // pointer directly. Otherwise, we must convert to avoid stack corruption.
  if constexpr (sizeof(timespec) == sizeof(__kernel_timespec)) {
    ret = syscall_impl<int>(SYS_utimensat_time64, dirfd, path, times, flags);
  } else {
    if (times != nullptr) {
      __kernel_timespec ts64[2]{to_kernel_timespec(times[0]),
                                to_kernel_timespec(times[1])};
      ret = syscall_impl<int>(SYS_utimensat_time64, dirfd, path, ts64, flags);
    } else {
      ret =
          syscall_impl<int>(SYS_utimensat_time64, dirfd, path, nullptr, flags);
    }
  }
#elif defined(SYS_utimensat)
  static_assert(
      sizeof(timespec::tv_nsec) == sizeof(long),
      "This legacy syscall fallback is only safe on platforms where tv_nsec "
      "matches the register size (long). It is unsafe on 32-bit platforms "
      "with 64-bit tv_nsec.");
  int ret = syscall_impl<int>(SYS_utimensat, dirfd, path, times, flags);
#else
#error "utimensat or utimensat_time64 syscalls not available."
#endif

  if (ret < 0)
    return Error(-static_cast<int>(ret));
  return ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_UTIMENSAT_H
