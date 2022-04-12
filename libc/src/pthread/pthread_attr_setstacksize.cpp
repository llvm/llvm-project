//===-- Implementation of the pthread_attr_setstacksize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_setstacksize.h"

#include "src/__support/common.h"

#include <errno.h>
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_setstacksize,
                   (pthread_attr_t *__restrict attr, size_t stacksize)) {
  if (stacksize < PTHREAD_STACK_MIN)
    return EINVAL;
  attr->__stack = nullptr;
  attr->__stacksize = stacksize;
  return 0;
}

} // namespace __llvm_libc
