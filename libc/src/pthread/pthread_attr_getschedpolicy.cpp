//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of pthread_attr_getschedpolicy.
///
//===----------------------------------------------------------------------===//

#include "pthread_attr_getschedpolicy.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_getschedpolicy,
                   (const pthread_attr_t *__restrict attr,
                    int *__restrict policy)) {
  LIBC_CRASH_ON_NULLPTR(attr);
  LIBC_CRASH_ON_NULLPTR(policy);

  *policy = attr->__schedpolicy;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
