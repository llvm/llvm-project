//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for exit_group.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_EXIT_GROUP_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_EXIT_GROUP_H

#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

[[noreturn]] LIBC_INLINE void exit_group(int status) {
  LIBC_SANITIZER_SYSCALL_PRE(exit_group, status);
  [[maybe_unused]] auto syscall_ret =
      syscall_impl<long>(SYS_exit_group, status);
  LIBC_SANITIZER_SYSCALL_POST(exit_group, static_cast<long>(syscall_ret),
                              status);
  __builtin_trap();
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_EXIT_GROUP_H
