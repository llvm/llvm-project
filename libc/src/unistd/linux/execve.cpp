//===-- Linux implementation of execve ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/execve.h"
#include "src/unistd/environ.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, execve,
                   (const char *path, char *const argv[], char *const envp[])) {
  long ret = __llvm_libc::syscall_impl(SYS_execve, path, argv, envp);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }

  // Control will not reach here on success but have a return statement will
  // keep the compilers happy.
  return ret;
}

} // namespace __llvm_libc
