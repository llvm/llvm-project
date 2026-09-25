//===-- Implementation of pthread_cond_clockwait --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_clockwait.h"

#include "pthread_cond_utils.h"

#include "hdr/errno_macros.h" // EINVAL
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_cond_clockwait,
                   (pthread_cond_t *__restrict cond,
                    pthread_mutex_t *__restrict mutex, clockid_t clock_id,
                    const struct timespec *__restrict abstime)) {
  CndVar *cndvar = pthread_cond_utils::to_cndvar(cond);
  Mutex *m = pthread_cond_utils::to_mutex(mutex);
  if (!pthread_cond_utils::is_supported_clock(clock_id))
    return EINVAL;
  return pthread_cond_utils::timed_wait(
      cndvar, m, abstime, pthread_cond_utils::is_realtime_clock(clock_id));
}

} // namespace LIBC_NAMESPACE_DECL
