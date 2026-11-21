//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for setreuid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETREUID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETREUID_H

#include "hdr/types/uid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> setreuid(uid_t ruid, uid_t euid) {
#if defined(SYS_setreuid32)
  LIBC_SANITIZER_SYSCALL_PRE(setreuid32, ruid, euid);
  auto syscall_ret = syscall_checked<int>(SYS_setreuid32, ruid, euid);
  LIBC_SANITIZER_SYSCALL_POST(setreuid32, syscall_result_to_long(syscall_ret),
                              ruid, euid);
  return syscall_ret;
#elif defined(SYS_setreuid)
  LIBC_SANITIZER_SYSCALL_PRE(setreuid, ruid, euid);
  auto syscall_ret = syscall_checked<int>(SYS_setreuid, ruid, euid);
  LIBC_SANITIZER_SYSCALL_POST(setreuid, syscall_result_to_long(syscall_ret),
                              ruid, euid);
  return syscall_ret;
#else
#error "SYS_setreuid and SYS_setreuid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETREUID_H
