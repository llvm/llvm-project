//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for inotify_init.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_INIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_INIT_H

#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> inotify_init(void) {
#ifdef SYS_inotify_init
  LIBC_SANITIZER_SYSCALL_PRE_0(inotify_init);
  auto syscall_ret = syscall_checked<int>(SYS_inotify_init);
  LIBC_SANITIZER_SYSCALL_POST(inotify_init,
                              syscall_result_to_long(syscall_ret));
  return syscall_ret;
#elif defined(SYS_inotify_init1)
  LIBC_SANITIZER_SYSCALL_PRE(inotify_init1, 0);
  auto syscall_ret = syscall_checked<int>(SYS_inotify_init1, 0);
  LIBC_SANITIZER_SYSCALL_POST(inotify_init1,
                              syscall_result_to_long(syscall_ret), 0);
  return syscall_ret;
#else
#error                                                                         \
    "inotify_init and inotify_init1 are unavailable. Unable to build inotify_init."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_INOTIFY_INIT_H
