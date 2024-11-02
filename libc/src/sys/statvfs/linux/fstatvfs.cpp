//===-- Linux implementation of fstatvfs ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/statvfs/fstatvfs.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/sys/statvfs/linux/statfs_utils.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, fstatvfs, (int fd, struct statvfs *buf)) {
  using namespace statfs_utils;
  cpp::optional<LinuxStatFs> result = linux_fstatfs(fd);
  if (result) {
    LIBC_ASSERT(buf != nullptr);
    *buf = statfs_to_statvfs(*result);
  }
  return result ? 0 : -1;
}

} // namespace LIBC_NAMESPACE
