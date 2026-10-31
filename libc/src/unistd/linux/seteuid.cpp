//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of seteuid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/seteuid.h"

#include "hdr/types/uid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/setresuid.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, seteuid, (uid_t uid)) {
  // TODO: Implement setxid propagation across threads.
  auto ret = linux_syscalls::setresuid(static_cast<uid_t>(-1), uid,
                                       static_cast<uid_t>(-1));
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
