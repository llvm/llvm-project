//===-- Implementation of the pthread_attr_getguardsize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_getguardsize.h"

#include "src/__support/common.h"

#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_attr_getguardsize,
                   (const pthread_attr_t *__restrict attr,
                    size_t *__restrict guardsize)) {
  *guardsize = attr->__guardsize;
  return 0;
}

} // namespace LIBC_NAMESPACE
