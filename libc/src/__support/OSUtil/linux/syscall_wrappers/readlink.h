//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for readlink.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_READLINK_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_READLINK_H

#include "hdr/fcntl_macros.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<ssize_t> readlink(const char *path, char *buf,
                                      size_t bufsiz) {
#ifdef SYS_readlinkat
  LIBC_SANITIZER_SYSCALL_PRE(readlinkat, AT_FDCWD, path, buf, bufsiz);
  auto syscall_ret =
      syscall_checked<ssize_t>(SYS_readlinkat, AT_FDCWD, path, buf, bufsiz);
  LIBC_SANITIZER_SYSCALL_POST(readlinkat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, path, buf, bufsiz);
  return syscall_ret;
#elif defined(SYS_readlink)
  LIBC_SANITIZER_SYSCALL_PRE(readlink, path, buf, bufsiz);
  auto syscall_ret = syscall_checked<ssize_t>(SYS_readlink, path, buf, bufsiz);
  LIBC_SANITIZER_SYSCALL_POST(readlink, syscall_result_to_long(syscall_ret),
                              path, buf, bufsiz);
  return syscall_ret;
#else
#error "readlink and readlinkat syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_READLINK_H
