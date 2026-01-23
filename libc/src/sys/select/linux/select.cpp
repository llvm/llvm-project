//===-- Linux implementation of select ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/select/select.h"

#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <stddef.h>      // For size_t
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

struct pselect6_sigset_t {
  sigset_t *ss;
  size_t ss_len;
};

LLVM_LIBC_FUNCTION(int, select,
                   (int nfds, fd_set *__restrict read_set,
                    fd_set *__restrict write_set, fd_set *__restrict error_set,
                    struct timeval *__restrict timeout)) {
  // Linux has a SYS_select syscall but it is not available on all
  // architectures. So, we use the SYS_pselect6 syscall which is more
  // widely available. However, SYS_pselect6 takes a struct timespec argument
  // instead of a struct timeval argument. Also, it takes an additional
  // argument which is a pointer to an object of a type defined above as
  // "pselect6_sigset_t".
  struct timespec ts {
    0, 0
  };
  if (timeout != nullptr) {
    // In general, if the tv_sec and tv_usec in |timeout| are correctly set,
    // then converting tv_usec to nanoseconds will not be a problem. However,
    // if tv_usec in |timeout| is more than a second, it can lead to overflows.
    // So, we detect such cases and adjust.
    constexpr time_t TIME_MAX = cpp::numeric_limits<time_t>::max();
    if ((TIME_MAX - timeout->tv_sec) < (timeout->tv_usec / 1000000)) {
      ts.tv_sec = TIME_MAX;
      ts.tv_nsec = 999999999;
    } else {
      ts.tv_sec = timeout->tv_sec + timeout->tv_usec / 1000000;
      ts.tv_nsec = timeout->tv_usec * 1000;
    }
  }
  pselect6_sigset_t pss{nullptr, sizeof(sigset_t)};
#if SYS_pselect6
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_pselect6, nfds, read_set,
                                              write_set, error_set, &ts, &pss);
#elif defined(SYS_pselect6_time64)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_pselect6_time64, nfds, read_set, write_set, error_set, &ts, &pss);
#else
#error "SYS_pselect6 and SYS_pselect6_time64 syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
