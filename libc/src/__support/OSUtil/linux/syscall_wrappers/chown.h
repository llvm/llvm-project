//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for chown.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_CHOWN_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_CHOWN_H

#include "hdr/fcntl_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/uid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> chown(const char *path, uid_t owner, gid_t group) {
#ifdef SYS_chown
  LIBC_SANITIZER_SYSCALL_PRE(chown, path, owner, group);
  auto syscall_ret = syscall_checked<int>(SYS_chown, path, owner, group);
  LIBC_SANITIZER_SYSCALL_POST(chown, syscall_result_to_long(syscall_ret), path,
                              owner, group);
  return syscall_ret;
#elif defined(SYS_fchownat)
  LIBC_SANITIZER_SYSCALL_PRE(fchownat, AT_FDCWD, path, owner, group, 0);
  auto syscall_ret =
      syscall_checked<int>(SYS_fchownat, AT_FDCWD, path, owner, group, 0);
  LIBC_SANITIZER_SYSCALL_POST(fchownat, syscall_result_to_long(syscall_ret),
                              AT_FDCWD, path, owner, group, 0);
  return syscall_ret;
#else
#error "chown and fchownat syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_CHOWN_H
