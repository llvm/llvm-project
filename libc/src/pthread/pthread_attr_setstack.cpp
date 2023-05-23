//===-- Implementation of the pthread_attr_setstack -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_setstack.h"
#include "pthread_attr_setstacksize.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h" // For STACK_ALIGNMENT

#include <errno.h>
#include <pthread.h>
#include <stdint.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_setstack,
                   (pthread_attr_t *__restrict attr, void *stack,
                    size_t stacksize)) {
  uintptr_t stackaddr = reinterpret_cast<uintptr_t>(stack);
  // TODO: Do we need to check for overflow on stackaddr + stacksize?
  if ((stackaddr % STACK_ALIGNMENT != 0) ||
      ((stackaddr + stacksize) % STACK_ALIGNMENT != 0))
    return EINVAL;

  if (stacksize < PTHREAD_STACK_MIN)
    return EINVAL;

  attr->__stack = stack;
  attr->__stacksize = stacksize;
  return 0;
}

} // namespace __llvm_libc
