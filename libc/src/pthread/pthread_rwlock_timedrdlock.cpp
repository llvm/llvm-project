//===-- Implementation of the Rwlock's timedrdlock function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_rwlock_timedrdlock.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/linux/rwlock.h"
#include "src/__support/time/linux/abs_timeout.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(
    sizeof(RwLock) == sizeof(pthread_rwlock_t) &&
        alignof(RwLock) == alignof(pthread_rwlock_t),
    "The public pthread_rwlock_t type must be of the same size and alignment "
    "as the internal rwlock type.");

LLVM_LIBC_FUNCTION(int, pthread_rwlock_timedrdlock,
                   (pthread_rwlock_t * rwlock,
                    const struct timespec *abstime)) {
  if (!rwlock)
    return EINVAL;
  RwLock *rw = reinterpret_cast<RwLock *>(rwlock);
  LIBC_ASSERT(abstime && "timedrdlock called with a null timeout");
  auto timeout =
      internal::AbsTimeout::from_timespec(*abstime, /*is_realtime=*/true);
  if (LIBC_LIKELY(timeout.has_value()))
    return static_cast<int>(rw->read_lock(timeout.value()));

  switch (timeout.error()) {
  case internal::AbsTimeout::Error::Invalid:
    return EINVAL;
  case internal::AbsTimeout::Error::BeforeEpoch:
    return ETIMEDOUT;
  }
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
