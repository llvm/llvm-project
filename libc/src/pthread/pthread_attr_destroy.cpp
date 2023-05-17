//===-- Implementation of the pthread_attr_destroy ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_destroy.h"

#include "src/__support/common.h"

#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_destroy, (pthread_attr_t *)) {
  // There is nothing to cleanup.
  return 0;
}

} // namespace __llvm_libc
