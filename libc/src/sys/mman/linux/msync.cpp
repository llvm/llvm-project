//===---------- Linux implementation of the msync function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/msync.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, msync, (void *addr, size_t len, int flags)) {
  long ret = syscall_impl(SYS_msync, cpp::bit_cast<long>(addr), len, flags);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return 0;
}
} // namespace LIBC_NAMESPACE_DECL
