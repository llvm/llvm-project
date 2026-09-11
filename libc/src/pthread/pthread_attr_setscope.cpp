//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of pthread_attr_setscope.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_attr_setscope.h"
#include "hdr/errno_macros.h"
#include "hdr/pthread_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_setscope,
                   ([[maybe_unused]] pthread_attr_t *attr,
                    int contentionscope)) {
  LIBC_CRASH_ON_NULLPTR(attr);

  switch (contentionscope) {
  case PTHREAD_SCOPE_SYSTEM:
    return 0;
  case PTHREAD_SCOPE_PROCESS:
    return ENOTSUP;
  default:
    return EINVAL;
  }
}

} // namespace LIBC_NAMESPACE_DECL

