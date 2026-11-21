//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ErrorOr-returning syscall wrapper for mkdir.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIR_H

#include "hdr/fcntl_macros.h"
#include "hdr/types/mode_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> mkdir(const char *path, mode_t mode) {
#ifdef SYS_mkdirat
  LIBC_SANITIZER_SYSCALL_PRE(mkdirat, AT_FDCWD, path, mode);
  auto syscall_ret = syscall_checked<int>(SYS_mkdirat, AT_FDCWD, path, mode);
  LIBC_SANITIZER_SYSCALL_POST(mkdirat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, path, mode);
  return syscall_ret;
#else
  LIBC_SANITIZER_SYSCALL_PRE(mkdir, path, mode);
  auto syscall_ret = syscall_checked<int>(SYS_mkdir, path, mode);
  LIBC_SANITIZER_SYSCALL_POST(mkdir, syscall_result_to_long(syscall_ret), path,
                              mode);
  return syscall_ret;
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIR_H
