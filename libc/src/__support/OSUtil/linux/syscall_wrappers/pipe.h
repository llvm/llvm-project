//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for pipe.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PIPE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PIPE_H

#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> pipe(int pipefd[2]) {
#ifdef SYS_pipe
  LIBC_SANITIZER_SYSCALL_PRE(pipe, pipefd);
  auto syscall_ret = syscall_checked<int>(SYS_pipe, pipefd);
  LIBC_SANITIZER_SYSCALL_POST(pipe, syscall_result_to_long(syscall_ret),
                              pipefd);
  return syscall_ret;
#elif defined(SYS_pipe2)
  LIBC_SANITIZER_SYSCALL_PRE(pipe2, pipefd, 0);
  auto syscall_ret = syscall_checked<int>(SYS_pipe2, pipefd, 0);
  LIBC_SANITIZER_SYSCALL_POST(pipe2, syscall_result_to_long(syscall_ret),
                              pipefd, 0);
  return syscall_ret;
#else
#error "pipe and pipe2 syscalls are not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_PIPE_H
