//===-- Implementation of the pthread_rwlockattr_init ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_rwlockattr_init.h"

#include "src/__support/common.h"

#include <pthread.h> // pthread_rwlockattr_t, PTHREAD_PROCESS_PRIVATE

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_rwlockattr_init,
                   (pthread_rwlockattr_t * attr)) {
  attr->pshared = PTHREAD_PROCESS_PRIVATE;
  attr->pref = PTHREAD_RWLOCK_PREFER_READER_NP;
  return 0;
}

} // namespace LIBC_NAMESPACE
