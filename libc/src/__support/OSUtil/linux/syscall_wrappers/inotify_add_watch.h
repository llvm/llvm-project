//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for inotify_add_watch.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_ADD_WATCH_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_ADD_WATCH_H

#include "hdr/stdint_proxy.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> inotify_add_watch(int fd, const char *pathname,
                                           uint32_t mask) {
  LIBC_SANITIZER_SYSCALL_PRE(inotify_add_watch, fd, pathname, mask);
  auto syscall_ret =
      syscall_checked<int>(SYS_inotify_add_watch, fd, pathname, mask);
  LIBC_SANITIZER_SYSCALL_POST(inotify_add_watch,
                              syscall_result_to_long(syscall_ret), fd, pathname,
                              mask);
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_ADD_WATCH_H
