//===-- Implementation of the pthread_rwlockattr_getkind_np ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_rwlockattr_getkind_np.h"

#include "src/__support/common.h"

#include <pthread.h> // pthread_rwlockattr_t

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_rwlockattr_getkind_np,
                   (const pthread_rwlockattr_t *__restrict attr,
                    int *__restrict pref)) {
  *pref = attr->pref;
  return 0;
}

} // namespace LIBC_NAMESPACE
