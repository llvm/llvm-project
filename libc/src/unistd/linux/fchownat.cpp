//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of fchownat.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/fchownat.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/fchownat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fchownat,
                   (int dirfd, const char *pathname, uid_t owner, gid_t group,
                    int flags)) {
  auto ret = linux_syscalls::fchownat(dirfd, pathname, owner, group, flags);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
