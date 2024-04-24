//===-- Implementation of the pthread_condattr_getpshared -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_condattr_getpshared.h"

#include "src/__support/common.h"

#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_condattr_getpshared,
                   (const pthread_condattr_t *__restrict attr,
                    int *__restrict pshared)) {
  *pshared = attr->pshared;
  return 0;
}

} // namespace LIBC_NAMESPACE
