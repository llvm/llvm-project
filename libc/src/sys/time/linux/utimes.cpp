//===-- Linux implementation of utimes ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/time/utimes.h"

#include "hdr/fcntl_macros.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_timeval.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, utimes,
                   (const char *path, const struct timeval times[2])) {
  int ret;

#ifdef SYS_utimes
  // No need to define a timespec struct, use the syscall directly.
  ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_utimes, path, times);
#elif defined(SYS_utimensat) || defined(SYS_utimensat_time64)

#if defined(SYS_utimensat)
  constexpr auto UTIMES_SYSCALL_ID = SYS_utimensat;
#elif defined(SYS_utimensat_time64)
  constexpr auto UTIMES_SYSCALL_ID = SYS_utimensat_time64;
#endif

  // the utimensat syscall requires a timespec struct, not timeval.
  struct timespec ts[2];
  struct timespec *ts_ptr = nullptr; // default value if times is nullptr

  // convert the microsec values in timeval struct times
  // to nanosecond values in timespec struct ts
  if (times != nullptr) {

    // ensure consistent values
    if ((times[0].tv_usec < 0 || times[1].tv_usec < 0) ||
        (times[0].tv_usec >= 1000000 || times[1].tv_usec >= 1000000)) {
      libc_errno = EINVAL;
      return -1;
    }

    // set seconds in ts
    ts[0].tv_sec = times[0].tv_sec;
    ts[1].tv_sec = times[1].tv_sec;

    // convert u-seconds to nanoseconds
    ts[0].tv_nsec =
        static_cast<decltype(ts[0].tv_nsec)>(times[0].tv_usec * 1000);
    ts[1].tv_nsec =
        static_cast<decltype(ts[1].tv_nsec)>(times[1].tv_usec * 1000);

    ts_ptr = ts;
  }

  // If times was nullptr, ts_ptr remains nullptr, which utimensat interprets
  // as setting times to the current time.

  // utimensat syscall.
  // flags=0 means don't follow symlinks (like utimes)
  ret = LIBC_NAMESPACE::syscall_impl<int>(UTIMES_SYSCALL_ID, AT_FDCWD, path,
                                          ts_ptr, 0);

#else
#error "utimes, utimensat, utimensat_time64,  syscalls not available."
#endif // SYS_utimensat

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return 0;
}
} // namespace LIBC_NAMESPACE_DECL
