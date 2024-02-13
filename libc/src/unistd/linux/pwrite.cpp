//===-- Linux implementation of pwrite ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pwrite.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <stdint.h>      // For uint64_t.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(ssize_t, pwrite,
                   (int fd, const void *buf, size_t count, off_t offset)) {
#ifdef LIBC_TARGET_ARCH_IS_RISCV32
  static_assert(sizeof(off_t) == 8);
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_pwrite64, fd, buf, count, (long)offset,
      (long)(((uint64_t)(offset)) >> 32));
#else
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_pwrite64, fd, buf,
                                                      count, offset);
#endif
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
