//===-- Linux implementation of ftruncate ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/ftruncate.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.
#include <unistd.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, ftruncate, (int fd, off_t len)) {
  int ret = __llvm_libc::syscall_impl(SYS_ftruncate, fd, len);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
