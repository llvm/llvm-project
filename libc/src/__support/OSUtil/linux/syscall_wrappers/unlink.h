//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for unlink.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_UNLINK_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_UNLINK_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> unlink(const char *path) {
#ifdef SYS_unlinkat
  LIBC_SANITIZER_SYSCALL_PRE(unlinkat, AT_FDCWD, path, 0);
  auto syscall_ret = syscall_checked<int>(SYS_unlinkat, AT_FDCWD, path, 0);
  LIBC_SANITIZER_SYSCALL_POST(unlinkat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, path, 0);
  return syscall_ret;
#elif defined(SYS_unlink)
  LIBC_SANITIZER_SYSCALL_PRE(unlink, path);
  auto syscall_ret = syscall_checked<int>(SYS_unlink, path);
  LIBC_SANITIZER_SYSCALL_POST(unlink, syscall_result_to_long(syscall_ret),
                              path);
  return syscall_ret;
#else
#error "unlink and unlinkat syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_UNLINK_H
