//===-- Linux implementation of lseek -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/lseek.h"
#include "src/errno/libc_errno.h"

#include "src/__support/File/linux/lseekImpl.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <sys/syscall.h> // For syscall numbers.
#include <unistd.h>      // For off_t.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(off_t, lseek, (int fd, off_t offset, int whence)) {
  auto result = internal::lseekimpl(fd, offset, whence);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE
