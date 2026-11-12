//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of madvise.
///
//===----------------------------------------------------------------------===//

#include "src/sys/mman/madvise.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/madvise.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// This function is currently linux only. It has to be refactored suitably if
// madvise is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(int, madvise, (void *addr, size_t size, int advice)) {
  auto result = linux_syscalls::madvise(addr, size, advice);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
