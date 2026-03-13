//===-- Implementation of the pthread_attr_setguardsize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_setguardsize.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <linux/param.h> // For EXEC_PAGESIZE.
#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_setguardsize,
                   (pthread_attr_t *__restrict attr, size_t guardsize)) {
  if (guardsize % EXEC_PAGESIZE != 0)
    return EINVAL;
  attr->__guardsize = guardsize;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
