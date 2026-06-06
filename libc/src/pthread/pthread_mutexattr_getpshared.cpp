//===-- Implementation of the pthread_mutexattr_getpshared ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_getpshared.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_getpshared,
                   (const pthread_mutexattr_t *__restrict attr,
                    int *__restrict pshared)) {
  *pshared = (*attr & unsigned(PThreadMutexAttrPos::PSHARED_MASK)) >>
             unsigned(PThreadMutexAttrPos::PSHARED_SHIFT);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
