//===-- Linux implementation of read --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/read.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/read.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, read, (int fd, void *buf, size_t count)) {
  auto result = linux_syscalls::read(fd, buf, count);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  // The cast is important since there is a check that dereferences the pointer
  // which fails on void*.
  MSAN_UNPOISON(reinterpret_cast<char *>(buf), count);
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
