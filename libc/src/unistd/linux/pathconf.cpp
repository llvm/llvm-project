//===-- Linux implementation of pathconf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON
#include "src/errno/libc_errno.h"
#include "src/unistd/pathconf_utils.h"

#include <stdint.h> // For uint64_t.
#include <sys/fstatvfs.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long, pathconf, (char *path, int name)) {
  if (cpp::optional<LinuxStatFs> result = linux_statfs(const char *path);) {
    return pathconfig(result.value(), name);
  }
  return -1;
}

} // namespace LIBC_NAMESPACE
