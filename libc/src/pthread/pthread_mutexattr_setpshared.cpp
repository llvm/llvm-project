//===-- Implementation of the pthread_mutexattr_setpshared ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_setpshared.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"

#include <errno.h>
#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_setpshared,
                   (pthread_mutexattr_t *__restrict attr, int pshared)) {
  if (pshared != PTHREAD_PROCESS_PRIVATE && pshared != PTHREAD_PROCESS_SHARED)
    return EINVAL;
  pthread_mutexattr_t old = *attr;
  old &= ~unsigned(PThreadMutexAttrPos::PSHARED_MASK);
  *attr = old | (pshared << unsigned(PThreadMutexAttrPos::PSHARED_SHIFT));
  return 0;
}

} // namespace LIBC_NAMESPACE
