//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of lsetxattr.
///
//===----------------------------------------------------------------------===//

#include "src/sys/xattr/lsetxattr.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/lsetxattr.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, lsetxattr,
                   (const char *path, const char *name, const void *value,
                    size_t size, int flags)) {
  LIBC_CRASH_ON_NULLPTR(path);
  LIBC_CRASH_ON_NULLPTR(name);
  if (size != 0)
    LIBC_CRASH_ON_NULLPTR(value);

  ErrorOr<int> ret = linux_syscalls::lsetxattr(path, name, value, size, flags);
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
