//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of timer_settime function.
///
//===----------------------------------------------------------------------===//

#include "src/time/timer_settime.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/timer_settime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, timer_settime,
                   (timer_t timerid, int flags,
                    const itimerspec *__restrict val,
                    itimerspec *__restrict old)) {
  auto result = linux_syscalls::timer_settime(timerid, flags, val, old);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
