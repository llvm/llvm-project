//===-- Implementation of the pthread_cond_init function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_init.h"

#include "include/llvm-libc-macros/pthread-macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/CndVar.h"

#include "hdr/errno_macros.h" // EINVAL
#include "hdr/time_macros.h"  // CLOCK_MONOTONIC, CLOCK_REALTIME

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_cond_init,
                   (pthread_cond_t *__restrict cond,
                    const pthread_condattr_t *__restrict attr)) {
  LIBC_CRASH_ON_NULLPTR(cond);
  // POSIX.1 says that CLOCK_REALTIME shall be used if the clock is not
  // monotonic explicitly.
  pthread_condattr_t condattr{
      /*clock=*/CLOCK_REALTIME,
      /*pshared=*/PTHREAD_PROCESS_PRIVATE,
  };
  if (attr)
    condattr = *attr;

  bool is_shared;
  switch (condattr.pshared) {
  case PTHREAD_PROCESS_PRIVATE:
    is_shared = false;
    break;
  case PTHREAD_PROCESS_SHARED:
    is_shared = true;
    break;
  default:
    return EINVAL;
  }

  bool is_realtime;
  switch (condattr.clock) {
  case CLOCK_MONOTONIC:
    is_realtime = false;
    break;
  case CLOCK_REALTIME:
    is_realtime = true;
    break;
  default:
    return EINVAL;
  }

  new (cond) CndVar(is_shared, is_realtime);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
