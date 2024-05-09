//===-- Implementation of the pthread_condattr_getclock -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_condattr_getclock.h"

#include "src/__support/common.h"

#include <pthread.h>   // pthread_condattr_t
#include <sys/types.h> // clockid_t

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_condattr_getclock,
                   (const pthread_condattr_t *__restrict attr,
                    clockid_t *__restrict clock_id)) {
  *clock_id = attr->clock;
  return 0;
}

} // namespace LIBC_NAMESPACE
