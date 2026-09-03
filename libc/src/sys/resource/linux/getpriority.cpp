//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getpriority.
///
//===----------------------------------------------------------------------===//

#include "src/sys/resource/getpriority.h"

#include "hdr/types/id_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpriority.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpriority, (int which, id_t who)) {
  auto result = linux_syscalls::getpriority(which, who);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  // The syscall returns (20 - nice), but we must return nice itself.
  return 20 - result.value();
}

} // namespace LIBC_NAMESPACE_DECL
