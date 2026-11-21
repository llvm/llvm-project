//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for link.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_LINK_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_LINK_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> link(const char *oldpath, const char *newpath) {
#ifdef SYS_linkat
  LIBC_SANITIZER_SYSCALL_PRE(linkat, AT_FDCWD, oldpath, AT_FDCWD, newpath, 0);
  auto syscall_ret =
      syscall_checked<int>(SYS_linkat, AT_FDCWD, oldpath, AT_FDCWD, newpath, 0);
  LIBC_SANITIZER_SYSCALL_POST(linkat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, oldpath, AT_FDCWD, newpath, 0);
  return syscall_ret;
#elif defined(SYS_link)
  LIBC_SANITIZER_SYSCALL_PRE(link, oldpath, newpath);
  auto syscall_ret = syscall_checked<int>(SYS_link, oldpath, newpath);
  LIBC_SANITIZER_SYSCALL_POST(link, syscall_result_to_long(syscall_ret),
                              oldpath, newpath);
  return syscall_ret;
#else
#error "link and linkat syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_LINK_H
