//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the pthread_mutex_timedlock function.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_mutex_timedlock.h"

#include "hdr/errno_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/abs_timeout.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_mutex_timedlock,
                   (pthread_mutex_t *__restrict mutex,
                    const struct timespec *__restrict abstime)) {
  LIBC_CRASH_ON_NULLPTR(abstime);
  auto *m = reinterpret_cast<Mutex *>(mutex);

  // According to POSIX.1, CLOCK_REALTIME is used for timedlock.
  auto timeout =
      internal::AbsTimeout::from_timespec(*abstime, /*is_realtime=*/true);

  if (!timeout.has_value()) {
    switch (timeout.error()) {
    case internal::AbsTimeout::Error::BeforeEpoch:
      if (m->try_lock() == MutexError::NONE)
        return 0;
      return ETIMEDOUT;
    case internal::AbsTimeout::Error::Invalid:
      return EINVAL;
    }
    __builtin_unreachable();
  }

  auto err = m->timed_lock(timeout.value());
  switch (err) {
  case MutexError::NONE:
    return 0;
  case MutexError::TIMEOUT:
    return ETIMEDOUT;
  case MutexError::DEADLOCK:
    return EDEADLK;
  case MutexError::OVERFLOW:
    return EAGAIN;
  default:
    return EINVAL;
  }
}

} // namespace LIBC_NAMESPACE_DECL
