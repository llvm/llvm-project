//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of pthread_attr_setschedpolicy.
///
//===----------------------------------------------------------------------===//

#include "pthread_attr_setschedpolicy.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_setschedpolicy,
                   (pthread_attr_t * attr, int policy)) {
  LIBC_CRASH_ON_NULLPTR(attr);
  attr->__schedpolicy = policy;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
