//===-- Linux implementation of the cnd_timedwait function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/cnd_timedwait.h"

#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/CndVar.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/abs_timeout.h"

#include <threads.h> // cnd_t, mtx_t, thrd_error, thrd_success, thrd_timedout

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(CndVar) == sizeof(cnd_t));

LLVM_LIBC_FUNCTION(int, cnd_timedwait,
                   (cnd_t * cond, mtx_t *mtx,
                    const struct timespec *abs_time)) {
  LIBC_CRASH_ON_NULLPTR(cond);
  LIBC_CRASH_ON_NULLPTR(mtx);
  LIBC_CRASH_ON_NULLPTR(abs_time);

  auto timeout =
      internal::AbsTimeout::from_timespec(*abs_time, /*is_realtime=*/true);
  if (LIBC_UNLIKELY(!timeout.has_value()))
    return timeout.error() == internal::AbsTimeout::Error::Invalid
               ? thrd_error
               : thrd_timedout;

  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  Mutex *mutex = reinterpret_cast<Mutex *>(mtx);
  switch (cndvar->wait(mutex, timeout.value())) {
  case CndVar::Result::Success:
    return thrd_success;
  case CndVar::Result::Timeout:
    return thrd_timedout;
  case CndVar::Result::MutexError:
    return thrd_error;
  }
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
