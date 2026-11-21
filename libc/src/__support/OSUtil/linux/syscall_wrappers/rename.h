//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for rename.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RENAME_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RENAME_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> rename(const char *oldpath, const char *newpath) {
#ifdef SYS_renameat2
  return syscall_checked<int>(SYS_renameat2, AT_FDCWD, oldpath, AT_FDCWD,
                              newpath, 0);
#elif defined(SYS_renameat)
  LIBC_SANITIZER_SYSCALL_PRE(renameat, AT_FDCWD, oldpath, AT_FDCWD, newpath);
  auto syscall_ret =
      syscall_checked<int>(SYS_renameat, AT_FDCWD, oldpath, AT_FDCWD, newpath);
  LIBC_SANITIZER_SYSCALL_POST(renameat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, oldpath, AT_FDCWD, newpath);
  return syscall_ret;
#elif defined(SYS_rename)
  LIBC_SANITIZER_SYSCALL_PRE(rename, oldpath, newpath);
  auto syscall_ret = syscall_checked<int>(SYS_rename, oldpath, newpath);
  LIBC_SANITIZER_SYSCALL_POST(rename, syscall_result_to_long(syscall_ret),
                              oldpath, newpath);
  return syscall_ret;
#else
#error "rename, renameat and renameat2 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RENAME_H
