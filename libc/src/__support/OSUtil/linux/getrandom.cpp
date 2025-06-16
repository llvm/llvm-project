//===------------ Linux implementation of getrandom -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/getrandom.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ssize_t getrandom(void *buf, size_t buflen, unsigned int flags) {
  ssize_t ret =
      LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_getrandom, buf, buflen, flags);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
