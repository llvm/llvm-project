//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for symlink.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SYMLINK_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SYMLINK_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> symlink(const char *path1, const char *path2) {
#ifdef SYS_symlink
  LIBC_SANITIZER_SYSCALL_PRE(symlink, path1, path2);
  auto syscall_ret = syscall_checked<int>(SYS_symlink, path1, path2);
  LIBC_SANITIZER_SYSCALL_POST(symlink, syscall_result_to_long(syscall_ret),
                              path1, path2);
  return syscall_ret;
#elif defined(SYS_symlinkat)
  LIBC_SANITIZER_SYSCALL_PRE(symlinkat, path1, AT_FDCWD, path2);
  auto syscall_ret =
      syscall_checked<int>(SYS_symlinkat, path1, AT_FDCWD, path2);
  LIBC_SANITIZER_SYSCALL_POST(symlinkat, syscall_result_to_long(syscall_ret),
                              path1, AT_FDCWD, path2);
  return syscall_ret;
#else
#error "symlink or symlinkat syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SYMLINK_H
