//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the pthread_getschedparam function.
///
//===----------------------------------------------------------------------===//

#include "pthread_getschedparam.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getschedparam,
                   (pthread_t th, int *__restrict policy,
                    struct sched_param *__restrict param)) {
  LIBC_CRASH_ON_NULLPTR(policy);
  LIBC_CRASH_ON_NULLPTR(param);
  auto *thread = reinterpret_cast<Thread *>(&th);
  auto result = thread->getschedparam();
  if (!result.has_value())
    return result.error();

  *policy = result.value().policy;
  *param = result.value().param;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
