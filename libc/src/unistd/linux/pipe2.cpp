//===-- Linux implementation of pipe --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pipe2.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pipe2, (int pipefd[2], int flags)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_pipe2, reinterpret_cast<long>(pipefd), flags);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  MSAN_UNPOISON(pipefd, sizeof(int) * 2);
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
