//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of statfs.
///
//===----------------------------------------------------------------------===//

#include "src/sys/statfs/statfs.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/statfs.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, statfs, (const char *path, struct statfs *buf)) {
  auto error_or_ret = linux_syscalls::statfs(path, buf);
  if (!error_or_ret) {
    libc_errno = error_or_ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
