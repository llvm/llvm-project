//===-- Implementation of pthread_cond_timedwait --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_timedwait.h"

#include "pthread_cond_utils.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_cond_timedwait,
                   (pthread_cond_t *__restrict cond,
                    pthread_mutex_t *__restrict mutex,
                    const struct timespec *__restrict abstime)) {
  CndVar *cndvar = pthread_cond_utils::to_cndvar(cond);
  return pthread_cond_utils::timed_wait(
      cndvar, pthread_cond_utils::to_mutex(mutex), abstime,
      cndvar->default_clock_is_realtime());
}

} // namespace LIBC_NAMESPACE_DECL
