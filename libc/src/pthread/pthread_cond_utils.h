//===-- Shared pthread condition variable helpers ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_UTILS_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_UTILS_H

#include "hdr/errno_macros.h" // EINVAL, ETIMEDOUT
#include "hdr/time_macros.h"  // CLOCK_MONOTONIC, CLOCK_REALTIME
#include "include/llvm-libc-types/clockid_t.h"
#include "include/llvm-libc-types/pthread_cond_t.h"
#include "include/llvm-libc-types/pthread_mutex_t.h"
#include "include/llvm-libc-types/struct_timespec.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/CndVar.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/abs_timeout.h"

namespace LIBC_NAMESPACE_DECL {
namespace pthread_cond_utils {

static_assert(
    sizeof(CndVar) == sizeof(pthread_cond_t) &&
        alignof(CndVar) == alignof(pthread_cond_t),
    "The public pthread_cond_t type must be of the same size and alignment "
    "as the internal condition variable type.");

LIBC_INLINE CndVar *to_cndvar(pthread_cond_t *cond) {
  LIBC_CRASH_ON_NULLPTR(cond);
  // TODO: use cpp:start_lifetime_as once
  // https://github.com/llvm/llvm-project/pull/193326 is merged
  return reinterpret_cast<CndVar *>(cond);
}

LIBC_INLINE Mutex *to_mutex(pthread_mutex_t *mutex) {
  LIBC_CRASH_ON_NULLPTR(mutex);
  // TODO: use cpp:start_lifetime_as once
  // https://github.com/llvm/llvm-project/pull/193326 is merged
  Mutex *m = reinterpret_cast<Mutex *>(mutex);
  LIBC_ASSERT(!m->is_robust() && "Robust mutex not supported yet");
  return m;
}

LIBC_INLINE bool is_supported_clock(clockid_t clock_id) {
  return clock_id == CLOCK_MONOTONIC || clock_id == CLOCK_REALTIME;
}

LIBC_INLINE bool is_realtime_clock(clockid_t clock_id) {
  return clock_id == CLOCK_REALTIME;
}

LIBC_INLINE int wait(CndVar *cond, Mutex *mutex,
                     cpp::optional<CndVar::Timeout> timeout) {
  switch (cond->wait(mutex, timeout)) {
  case CndVarResult::Success:
    return 0;
  case CndVarResult::Timeout:
    return ETIMEDOUT;
  case CndVarResult::MutexError:
    return EINVAL;
  }
  __builtin_unreachable();
}

LIBC_INLINE int timed_wait(CndVar *cond, Mutex *mutex,
                           const struct timespec *abstime, bool is_realtime) {
  LIBC_CRASH_ON_NULLPTR(abstime);
  auto timeout =
      internal::AbsTimeout::from_timespec(*abstime, /*realtime=*/is_realtime);
  if (LIBC_LIKELY(timeout.has_value()))
    return wait(cond, mutex, timeout.value());

  switch (timeout.error()) {
  case internal::AbsTimeout::Error::Invalid:
    return EINVAL;
  case internal::AbsTimeout::Error::BeforeEpoch:
    return ETIMEDOUT;
  }
  __builtin_unreachable();
}

} // namespace pthread_cond_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_UTILS_H
