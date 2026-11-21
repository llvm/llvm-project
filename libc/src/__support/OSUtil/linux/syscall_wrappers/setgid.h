//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for setgid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGID_H

#include "hdr/types/gid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> setgid(gid_t gid) {
#if defined(SYS_setgid32)
  LIBC_SANITIZER_SYSCALL_PRE(setgid32, gid);
  auto syscall_ret = syscall_checked<int>(SYS_setgid32, gid);
  LIBC_SANITIZER_SYSCALL_POST(setgid32, syscall_result_to_long(syscall_ret),
                              gid);
  return syscall_ret;
#elif defined(SYS_setgid)
  LIBC_SANITIZER_SYSCALL_PRE(setgid, gid);
  auto syscall_ret = syscall_checked<int>(SYS_setgid, gid);
  LIBC_SANITIZER_SYSCALL_POST(setgid, syscall_result_to_long(syscall_ret), gid);
  return syscall_ret;
#else
#error "SYS_setgid and SYS_setgid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGID_H
