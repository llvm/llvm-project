//===-- Implementation of the pthread_attr_setstack -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_setstack.h"

#include "src/__support/common.h"

#include <errno.h>
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <pthread.h>
#include <stdint.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_setstack,
                   (pthread_attr_t *__restrict attr, void *stack,
                    size_t stacksize)) {
  if (stacksize < PTHREAD_STACK_MIN)
    return EINVAL;
  uintptr_t stackaddr = reinterpret_cast<uintptr_t>(stack);
  if ((stackaddr % 16 != 0) || ((stackaddr + stacksize) % 16 != 0))
    return EINVAL;
  attr->__stack = stack;
  attr->__stacksize = stacksize;
  return 0;
}

} // namespace __llvm_libc
