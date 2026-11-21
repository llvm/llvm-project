//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for dup2.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_DUP2_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_DUP2_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/OSUtil/linux/syscall_wrappers/fcntl.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> dup2(int oldfd, int newfd) {
#ifdef SYS_dup2
  LIBC_SANITIZER_SYSCALL_PRE(dup2, oldfd, newfd);
  auto syscall_ret = syscall_checked<int>(SYS_dup2, oldfd, newfd);
  LIBC_SANITIZER_SYSCALL_POST(dup2, syscall_result_to_long(syscall_ret), oldfd,
                              newfd);
  return syscall_ret;
#elif defined(SYS_dup3)
  if (oldfd == newfd) {
    auto ret = fcntl(oldfd, F_GETFD);
    if (!ret)
      return Error(ret.error());
    return oldfd;
  }
  LIBC_SANITIZER_SYSCALL_PRE(dup3, oldfd, newfd, 0);
  auto syscall_ret = syscall_checked<int>(SYS_dup3, oldfd, newfd, 0);
  LIBC_SANITIZER_SYSCALL_POST(dup3, syscall_result_to_long(syscall_ret), oldfd,
                              newfd, 0);
  return syscall_ret;
#else
#error "dup2 and dup3 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_DUP2_H
