//===-- Implementation of creat -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/creat.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include "hdr/fcntl_macros.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, creat, (const char *path, int mode_flags)) {
  ErrorOr<int> fd =
      linux_syscalls::open(path, O_CREAT | O_WRONLY | O_TRUNC, mode_flags);

  if (!fd) {
    libc_errno = fd.error();
    return -1;
  }
  return fd.value();
}

} // namespace LIBC_NAMESPACE_DECL
