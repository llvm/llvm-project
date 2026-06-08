//===-- Linux implementation of the cnd_timedwait function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/cnd_timedwait.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/CndVar.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/abs_timeout.h"

#include <threads.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(CndVar) == sizeof(cnd_t));
static_assert(sizeof(Mutex) == sizeof(mtx_t) &&
              alignof(Mutex) == alignof(mtx_t));

LLVM_LIBC_FUNCTION(int, cnd_timedwait,
                   (cnd_t *__restrict cond, mtx_t *__restrict mtx,
                    const struct timespec *__restrict time_point)) {
  LIBC_CRASH_ON_NULLPTR(time_point);
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  Mutex *mutex = reinterpret_cast<Mutex *>(mtx);

  // time_point is TIME_UTC-based, so we assume realtime clock here.
  auto timeout =
      internal::AbsTimeout::from_timespec(*time_point, /*realtime=*/true);

  if (!timeout.has_value()) {
    switch (timeout.error()) {
    case internal::AbsTimeout::Error::BeforeEpoch:
      return thrd_timedout;
    case internal::AbsTimeout::Error::Invalid:
      return thrd_error;
    }
    __builtin_unreachable();
  }

  switch (cndvar->wait(mutex, timeout.value())) {
  case CndVarResult::Success:
    return thrd_success;
  case CndVarResult::Timeout:
    return thrd_timedout;
  case CndVarResult::MutexError:
    return thrd_error;
  }
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
