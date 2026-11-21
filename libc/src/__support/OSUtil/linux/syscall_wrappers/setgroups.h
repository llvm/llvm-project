//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for setgroups.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGROUPS_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGROUPS_H

#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> setgroups(size_t size, const gid_t *list) {
  // Confusingly, 32-bit is the newer syscall, replacing the 16-bit legacy
  // version.
#if defined(SYS_setgroups32)
  LIBC_SANITIZER_SYSCALL_PRE(setgroups32, size, list);
  auto syscall_ret = syscall_checked<int>(SYS_setgroups32, size, list);
  LIBC_SANITIZER_SYSCALL_POST(setgroups32, syscall_result_to_long(syscall_ret),
                              size, list);
  return syscall_ret;
#elif defined(SYS_setgroups)
  LIBC_SANITIZER_SYSCALL_PRE(setgroups, size, list);
  auto syscall_ret = syscall_checked<int>(SYS_setgroups, size, list);
  LIBC_SANITIZER_SYSCALL_POST(setgroups, syscall_result_to_long(syscall_ret),
                              size, list);
  return syscall_ret;
#else
#error "SYS_setgroups and SYS_setgroups32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SETGROUPS_H
