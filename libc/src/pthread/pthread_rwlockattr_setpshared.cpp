//===-- Implementation of the pthread_rwlockattr_setpshared ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_rwlockattr_setpshared.h"

#include "src/__support/common.h"

#include <errno.h> // EINVAL
#include <pthread.h> // pthread_rwlockattr_t, PTHREAD_PROCESS_SHARED, PTHREAD_PROCESS_PRIVATE

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_rwlockattr_setpshared,
                   (pthread_rwlockattr_t * attr, int pshared)) {
  if (pshared != PTHREAD_PROCESS_SHARED && pshared != PTHREAD_PROCESS_PRIVATE)
    return EINVAL;

  attr->pshared = pshared;
  return 0;
}

} // namespace LIBC_NAMESPACE
