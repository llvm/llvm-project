//===-- Implementation of the pthread_mutexattr_init ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_destroy.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_destroy, (pthread_mutexattr_t *)) {
  return 0;
}

} // namespace __llvm_libc
