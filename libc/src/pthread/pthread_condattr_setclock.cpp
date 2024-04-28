//===-- Implementation of the pthread_condattr_setclock -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_condattr_setclock.h"

#include "src/__support/common.h"

#include <errno.h>     // EINVAL
#include <pthread.h>   // pthread_condattr_t
#include <sys/types.h> // clockid_t
#include <time.h>      // CLOCK_MONOTONIC, CLOCK_REALTIME

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_condattr_setclock,
                   (pthread_condattr_t * attr, clockid_t clock)) {

  if (clock != CLOCK_MONOTONIC && clock != CLOCK_REALTIME)
    return EINVAL;

  attr->clock = clock;
  return 0;
}

} // namespace LIBC_NAMESPACE
