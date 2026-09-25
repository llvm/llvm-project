//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of pthread_attr_init.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_attr_init.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/pthread/pthread_attr.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_init, (pthread_attr_t * attr)) {
  LIBC_CRASH_ON_NULLPTR(attr);

  *attr = DEFAULT_PTHREAD_ATTR;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
