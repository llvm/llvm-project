//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for getresuid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETRESUID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETRESUID_H

#include "hdr/types/uid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> getresuid(uid_t *ruid, uid_t *euid, uid_t *suid) {
#if defined(SYS_getresuid32)
  LIBC_SANITIZER_SYSCALL_PRE(getresuid32, ruid, euid, suid);
  auto syscall_ret = syscall_checked<int>(SYS_getresuid32, ruid, euid, suid);
  LIBC_SANITIZER_SYSCALL_POST(getresuid32, syscall_result_to_long(syscall_ret),
                              ruid, euid, suid);
  return syscall_ret;
#elif defined(SYS_getresuid)
  LIBC_SANITIZER_SYSCALL_PRE(getresuid, ruid, euid, suid);
  auto syscall_ret = syscall_checked<int>(SYS_getresuid, ruid, euid, suid);
  LIBC_SANITIZER_SYSCALL_POST(getresuid, syscall_result_to_long(syscall_ret),
                              ruid, euid, suid);
  return syscall_ret;
#else
#error "SYS_getresuid and SYS_getresuid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETRESUID_H
