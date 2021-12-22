//===-- Linux implementation of write -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/write.h"

#include "include/sys/syscall.h"          // For syscall numbers.
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(ssize_t, write, (int fd, const void *buf, size_t count)) {
  long ret = __llvm_libc::syscall(SYS_write, fd, buf, count);
  if (ret < 0) {
    llvmlibc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
