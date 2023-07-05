//===-- Implementation of nanosleep function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/nanosleep.h"

#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

// TODO(michaelrj): Move this into time/linux with the other syscalls.
LLVM_LIBC_FUNCTION(int, nanosleep,
                   (const struct timespec *req, struct timespec *rem)) {
  int ret = __llvm_libc::syscall_impl(SYS_nanosleep, req, rem);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
