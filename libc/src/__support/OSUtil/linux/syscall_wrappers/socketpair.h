//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for socketpair.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETPAIR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETPAIR_H

#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> socketpair(int domain, int type, int protocol,
                                    int sv[2]) {
  LIBC_SANITIZER_SYSCALL_PRE(socketpair, domain, type, protocol, sv);
  auto syscall_ret =
      syscall_checked<int>(SYS_socketpair, domain, type, protocol, sv);
  LIBC_SANITIZER_SYSCALL_POST(socketpair, syscall_result_to_long(syscall_ret),
                              domain, type, protocol, sv);
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETPAIR_H
