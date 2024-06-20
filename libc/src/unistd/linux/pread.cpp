//===-- Linux implementation of pread -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pread.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON
#include "src/errno/libc_errno.h"
#include <stdint.h>      // For uint64_t.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(ssize_t, pread,
                   (int fd, void *buf, size_t count, off_t offset)) {
#ifdef LIBC_TARGET_ARCH_IS_RISCV32
  static_assert(sizeof(off_t) == 8);
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_pread64, fd, buf, count, (long)offset,
      (long)(((uint64_t)(offset)) >> 32));
#else
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_pread64, fd, buf,
                                                      count, offset);
#endif
  // The cast is important since there is a check that dereferences the pointer
  // which fails on void*.
  MSAN_UNPOISON(reinterpret_cast<char *>(buf), count);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
