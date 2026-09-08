//===-- Linux implementation of fstatvfs ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/statvfs/fstatvfs.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fstatfs.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/sys/statvfs/linux/statfs_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fstatvfs, (int fd, struct statvfs *buf)) {
  LIBC_CRASH_ON_NULLPTR(buf);
  struct statfs result;
  auto error_or_ret = linux_syscalls::fstatfs(fd, &result);
  if (!error_or_ret) {
    libc_errno = error_or_ret.error();
    return -1;
  }
  *buf = statfs_utils::statfs_to_statvfs(result);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
