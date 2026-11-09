//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the mtx_timedlock function.
///
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_timedlock.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/abs_timeout.h"

#include <threads.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mtx_timedlock,
                   (mtx_t *__restrict mutex,
                    const struct timespec *__restrict time_point)) {
  LIBC_CRASH_ON_NULLPTR(time_point);
  auto *m = reinterpret_cast<Mutex *>(mutex);

  auto timeout =
      internal::AbsTimeout::from_timespec(*time_point, /*is_realtime=*/true);

  if (!timeout.has_value()) {
    switch (timeout.error()) {
    case internal::AbsTimeout::Error::BeforeEpoch:
      if (m->try_lock() == MutexError::NONE)
        return thrd_success;
      return thrd_timedout;
    case internal::AbsTimeout::Error::Invalid:
      return thrd_error;
    }
    __builtin_unreachable();
  }

  auto err = m->timed_lock(timeout.value());
  switch (err) {
  case MutexError::NONE:
    return thrd_success;
  case MutexError::TIMEOUT:
    return thrd_timedout;
  default:
    return thrd_error;
  }
}

} // namespace LIBC_NAMESPACE_DECL
