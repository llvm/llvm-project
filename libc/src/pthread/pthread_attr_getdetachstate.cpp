//===-- Implementation of the pthread_attr_getdetachstate -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_getdetachstate.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_getdetachstate,
                   (const pthread_attr_t *attr, int *detachstate)) {
  *detachstate = attr->__detachstate;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
