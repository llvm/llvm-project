//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of renameat.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/renameat.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/renameat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, renameat,
                   (int olddirfd, const char *oldpath, int newdirfd,
                    const char *newpath)) {
  auto result = linux_syscalls::renameat(olddirfd, oldpath, newdirfd, newpath);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
