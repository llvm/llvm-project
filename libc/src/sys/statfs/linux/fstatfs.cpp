//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of fstatfs.
///
//===----------------------------------------------------------------------===//

#include "src/sys/statfs/fstatfs.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fstatfs.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fstatfs, (int fd, struct statfs *buf)) {
  auto error_or_ret = linux_syscalls::fstatfs(fd, buf);
  if (!error_or_ret) {
    libc_errno = error_or_ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
