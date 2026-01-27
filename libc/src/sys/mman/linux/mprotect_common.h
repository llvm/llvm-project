//===---------- Shared Linux implementation of POSIX mprotect. ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

namespace mprotect_common {

// This function is currently linux only. It has to be refactored suitably if
// mprotect is to be supported on non-linux operating systems also.
LIBC_INLINE ErrorOr<int> mprotect_impl(void *addr, size_t size, int prot) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_mprotect, reinterpret_cast<long>(addr), size, prot);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    return Error(-ret);
  }

  return 0;
}

} // namespace mprotect_common

} // namespace LIBC_NAMESPACE_DECL
