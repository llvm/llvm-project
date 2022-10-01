//===---------- Linux implementation of the POSIX posix_madvise function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/posix_madvise.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

// This function is currently linux only. It has to be refactored suitably if
// posix_madvise is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(int, posix_madvise, (void *addr, size_t size, int advice)) {
  // POSIX_MADV_DONTNEED does nothing because the default MADV_DONTNEED may
  // cause data loss, which the posix madvise does not allow.
  if (advice == POSIX_MADV_DONTNEED) {
    return 0;
  }
  long ret_val = __llvm_libc::syscall_impl(
      SYS_madvise, reinterpret_cast<long>(addr), size, advice);
  return ret_val < 0 ? -ret_val : 0;
}

} // namespace __llvm_libc
