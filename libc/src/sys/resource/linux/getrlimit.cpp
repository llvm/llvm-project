//===-- Linux implementation of getrlimit ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/resource/getrlimit.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/resource.h> // For struct rlimit
#include <sys/syscall.h>  // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, getrlimit, (int res, struct rlimit *limits)) {
  long ret = __llvm_libc::syscall_impl(SYS_prlimit64, 0, res, nullptr, limits);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
