//===-- Implementation file for getitimer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/time/getitimer.h"
#include "hdr/types/struct_itimerval.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getitimer, (int which, struct itimerval *curr_value)) {
  long ret = 0;
  if constexpr (sizeof(time_t) > sizeof(long)) {
    // There is no SYS_getitimer_time64 call, so we can't use time_t directly.
    long curr_value32[4];
    ret =
        LIBC_NAMESPACE::syscall_impl<long>(SYS_getitimer, which, curr_value32);
    if (!ret) {
      curr_value->it_interval.tv_sec = curr_value32[0];
      curr_value->it_interval.tv_usec = curr_value32[1];
      curr_value->it_value.tv_sec = curr_value32[2];
      curr_value->it_value.tv_usec = curr_value32[3];
    }
  } else {
    ret = LIBC_NAMESPACE::syscall_impl<long>(SYS_getitimer, which, curr_value);
  }

  // On failure, return -1 and set errno.
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
