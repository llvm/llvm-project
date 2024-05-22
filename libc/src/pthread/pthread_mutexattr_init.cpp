//===-- Implementation of the pthread_mutexattr_init ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_init.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_init, (pthread_mutexattr_t * attr)) {
  // Set the default attributes and mark the attribute object as initiliazed
  // by setting the first bit.
  *attr = DEFAULT_MUTEXATTR;
  return 0;
}

} // namespace LIBC_NAMESPACE
