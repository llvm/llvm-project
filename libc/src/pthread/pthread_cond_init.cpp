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

static_assert(
    sizeof(CndVar) == sizeof(pthread_cond_t) &&
        alignof(CndVar) == alignof(pthread_cond_t),
    "The public pthread_cond_t type must be of the same size and alignment "
    "as the internal condition variable type.");

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

  // POSIX.1 does not specify behavior for invalid clock values.
  bool is_shared = condattr.pshared == PTHREAD_PROCESS_SHARED;
  bool is_realtime = condattr.clock == CLOCK_REALTIME;
  new (cond) CndVar(is_shared, is_realtime);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
