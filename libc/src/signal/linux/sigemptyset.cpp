//===-- Linux implementation of sigemptyset -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigemptyset.h"
#include "include/errno.h" // For E* macros.
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sigemptyset, (sigset_t * set)) {
  if (!set) {
    llvmlibc_errno = EINVAL;
    return -1;
  }
  *set = __llvm_libc::Sigset::emptySet();
  return 0;
}

} // namespace __llvm_libc
