//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ErrorOr-returning syscall wrapper for mkdirat.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIRAT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIRAT_H

#include "hdr/types/mode_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> mkdirat(int dfd, const char *path, mode_t mode) {
  LIBC_SANITIZER_SYSCALL_PRE(mkdirat, dfd, path, mode);
  auto syscall_ret = syscall_checked<int>(SYS_mkdirat, dfd, path, mode);
  LIBC_SANITIZER_SYSCALL_POST(mkdirat, syscall_result_to_long(syscall_ret), dfd,
                              path, mode);
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_MKDIRAT_H
