//===-- Linux implementation of fpathconf ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/unistd/pathconf_utils.h"

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long, fpathconf, (int fd, int name)) {
  if (cpp::optional<LinuxStatFs> result = linux_fstatfs(fd))
    return pathconfig(result.value(), name);
  return -1;
}

} // namespace LIBC_NAMESPACE
