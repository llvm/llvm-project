//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of thrd_sleep.
///
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_sleep.h"

#include "hdr/errno_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/nanosleep.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, thrd_sleep,
                   (const struct timespec *duration,
                    struct timespec *remaining)) {
  LIBC_CRASH_ON_NULLPTR(duration);

  auto result = linux_syscalls::nanosleep(duration, remaining);
  if (!result) {
    if (result.error() == EINTR)
      return -1;
    return -2;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
