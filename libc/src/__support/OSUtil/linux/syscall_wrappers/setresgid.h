//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for setresgid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESGID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESGID_H

#include "hdr/types/gid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> setresgid(gid_t rgid, gid_t egid, gid_t sgid) {
#if defined(SYS_setresgid32)
  LIBC_SANITIZER_SYSCALL_PRE(setresgid32, rgid, egid, sgid);
  auto syscall_ret = syscall_checked<int>(SYS_setresgid32, rgid, egid, sgid);
  LIBC_SANITIZER_SYSCALL_POST(setresgid32, syscall_result_to_long(syscall_ret),
                              rgid, egid, sgid);
  return syscall_ret;
#elif defined(SYS_setresgid)
  LIBC_SANITIZER_SYSCALL_PRE(setresgid, rgid, egid, sgid);
  auto syscall_ret = syscall_checked<int>(SYS_setresgid, rgid, egid, sgid);
  LIBC_SANITIZER_SYSCALL_POST(setresgid, syscall_result_to_long(syscall_ret),
                              rgid, egid, sgid);
  return syscall_ret;
#else
#error "SYS_setresgid and SYS_setresgid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETRESGID_H
