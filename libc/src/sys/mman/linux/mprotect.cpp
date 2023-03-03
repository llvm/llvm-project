//===---------- Linux implementation of the POSIX mprotect function -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mprotect.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

// This function is currently linux only. It has to be refactored suitably if
// mprotect is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(int, mprotect, (void *addr, size_t size, int prot)) {
  long ret_val = __llvm_libc::syscall_impl(
      SYS_mprotect, reinterpret_cast<long>(addr), size, prot);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret_val < 0) {
    libc_errno = -ret_val;
    return -1;
  }

  return 0;
}

} // namespace __llvm_libc
