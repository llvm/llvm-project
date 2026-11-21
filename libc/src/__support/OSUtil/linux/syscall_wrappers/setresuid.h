//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for setresuid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESUID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESUID_H

#include "hdr/types/uid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> setresuid(uid_t ruid, uid_t euid, uid_t suid) {
#if defined(SYS_setresuid32)
  LIBC_SANITIZER_SYSCALL_PRE(setresuid32, ruid, euid, suid);
  auto syscall_ret = syscall_checked<int>(SYS_setresuid32, ruid, euid, suid);
  LIBC_SANITIZER_SYSCALL_POST(setresuid32, syscall_result_to_long(syscall_ret),
                              ruid, euid, suid);
  return syscall_ret;
#elif defined(SYS_setresuid)
  LIBC_SANITIZER_SYSCALL_PRE(setresuid, ruid, euid, suid);
  auto syscall_ret = syscall_checked<int>(SYS_setresuid, ruid, euid, suid);
  LIBC_SANITIZER_SYSCALL_POST(setresuid, syscall_result_to_long(syscall_ret),
                              ruid, euid, suid);
  return syscall_ret;
#else
#error "SYS_setresuid and SYS_setresuid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESUID_H
