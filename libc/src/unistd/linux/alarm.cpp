//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of alarm.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/alarm.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // For syscall numbers.

#ifndef SYS_alarm
#include "hdr/types/struct_itimerval.h"
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, alarm, (unsigned int seconds)) {
#ifdef SYS_alarm
  return static_cast<unsigned int>(
      LIBC_NAMESPACE::syscall_impl<long>(SYS_alarm, seconds));
#elif defined(SYS_setitimer)
  // On 32-bit architectures with 64-bit time_t, SYS_setitimer still expects
  // 32-bit fields. We must convert itimerval to use 32-bit fields.
  if constexpr (sizeof(time_t) > sizeof(long)) {
    long itv32[4] = {0, 0, static_cast<long>(seconds), 0};
    long old_itv32[4];
    long ret = LIBC_NAMESPACE::syscall_impl<long>(
        SYS_setitimer, 0 /* ITIMER_REAL */, itv32, old_itv32);
    if (ret < 0)
      return 0;
    return static_cast<unsigned int>(old_itv32[2] + (old_itv32[3] > 0 ? 1 : 0));
  } else {
    struct itimerval itv, old_itv;
    itv.it_interval.tv_sec = 0;
    itv.it_interval.tv_usec = 0;
    itv.it_value.tv_sec = seconds;
    itv.it_value.tv_usec = 0;
    long ret = LIBC_NAMESPACE::syscall_impl<long>(
        SYS_setitimer, 0 /* ITIMER_REAL */, &itv, &old_itv);
    if (ret < 0)
      return 0;
    return static_cast<unsigned int>(old_itv.it_value.tv_sec +
                                     (old_itv.it_value.tv_usec > 0 ? 1 : 0));
  }
#else
#error "alarm implementation not available for this architecture"
#endif
}

} // namespace LIBC_NAMESPACE_DECL
