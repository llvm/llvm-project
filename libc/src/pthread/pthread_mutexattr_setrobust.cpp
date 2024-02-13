//===-- Implementation of the pthread_mutexattr_setrobust -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_setrobust.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"

#include <errno.h>
#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_setrobust,
                   (pthread_mutexattr_t *__restrict attr, int robust)) {
  if (robust != PTHREAD_MUTEX_STALLED && robust != PTHREAD_MUTEX_ROBUST)
    return EINVAL;
  pthread_mutexattr_t old = *attr;
  old &= ~unsigned(PThreadMutexAttrPos::ROBUST_MASK);
  *attr = old | (robust << unsigned(PThreadMutexAttrPos::ROBUST_SHIFT));
  return 0;
}

} // namespace LIBC_NAMESPACE
