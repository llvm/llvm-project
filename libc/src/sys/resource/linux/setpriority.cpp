//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of setpriority.
///
//===----------------------------------------------------------------------===//

#include "src/sys/resource/setpriority.h"

#include "hdr/types/id_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/setpriority.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpriority, (int which, id_t who, int prio)) {
  auto result = linux_syscalls::setpriority(which, who, prio);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
