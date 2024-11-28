//===---------- Linux implementation of the mrelease function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/process_mrelease.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, process_mrelease, (int pidfd, unsigned int flags)) {
#ifdef SYS_process_mrelease
  long ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_process_mrelease, pidfd, flags);

  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }

  return 0;
#else
  // The system call is not available.
  (void)pidfd;
  (void)flags;
  libc_errno = ENOSYS;
  return -1;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
