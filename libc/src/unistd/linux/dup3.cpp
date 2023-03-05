//===-- Linux implementation of dup3 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/dup3.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, dup3, (int oldfd, int newfd, int flags)) {
  // If dup2 syscall is available, we make use of directly.
  long ret = __llvm_libc::syscall_impl(SYS_dup3, oldfd, newfd, flags);
  if (ret >= 0)
    return ret;
  libc_errno = -ret;
  return -1;
}

} // namespace __llvm_libc
