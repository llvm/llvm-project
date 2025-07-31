//===-- Implementation file for setitimer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/sys/time/setitimer.h"
#include "hdr/types/struct_itimerval.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setitimer,
                   (int which, const struct itimerval *new_value,
                    struct itimerval *old_value)) {
  long ret = 0;
  if constexpr (sizeof(time_t) > sizeof(long)) {
    // There is no SYS_setitimer_time64 call, so we can't use time_t directly,
    // and need to convert it to long first.
    long new_value32[4] = {static_cast<long>(new_value->it_interval.tv_sec),
                           static_cast<long>(new_value->it_interval.tv_usec),
                           static_cast<long>(new_value->it_value.tv_sec),
                           static_cast<long>(new_value->it_value.tv_usec)};
    long old_value32[4];

    ret = LIBC_NAMESPACE::syscall_impl<long>(SYS_setitimer, which, new_value32,
                                             old_value32);

    if (!ret && old_value) {
      old_value->it_interval.tv_sec = old_value32[0];
      old_value->it_interval.tv_usec = old_value32[1];
      old_value->it_value.tv_sec = old_value32[2];
      old_value->it_value.tv_usec = old_value32[3];
    }
  } else {
    ret = LIBC_NAMESPACE::syscall_impl<long>(SYS_setitimer, which, new_value,
                                             old_value);
  }

  // On failure, return -1 and set errno.
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
