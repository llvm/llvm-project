//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getgroups.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getgroups.h"

#include "hdr/types/gid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getgroups.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getgroups, (int size, gid_t *list)) {
  auto ret = linux_syscalls::getgroups(size, list);
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
