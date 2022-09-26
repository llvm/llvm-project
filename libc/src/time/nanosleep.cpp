//===-- Implementation of nanosleep function
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/nanosleep.h"
#include "include/sys/syscall.h"          // For syscall numbers.
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, nanosleep,
                   (const struct timespec *req, struct timespec *rem)) {
  int ret = __llvm_libc::syscall(SYS_nanosleep, req, rem);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
