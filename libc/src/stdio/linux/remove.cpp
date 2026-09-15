//===-- Linux implementation of remove ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/remove.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/unlinkat.h"
#include "src/__support/common.h"

#include "hdr/errno_macros.h" // For EISDIR.
#include "hdr/fcntl_macros.h" // For AT_* macros.
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, remove, (const char *path)) {
  // We first try unlinking it as a file. If it is ia file, it will succeed. If
  // it fails with EISDIR, we will try unlinking it as a directory.
  auto ret = linux_syscalls::unlinkat(AT_FDCWD, path, 0);
  if (!ret && ret.error() == EISDIR)
    ret = linux_syscalls::unlinkat(AT_FDCWD, path, AT_REMOVEDIR);
  if (ret)
    return 0;
  libc_errno = ret.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
