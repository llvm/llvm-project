//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of mlock2.
///
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mlock2.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/mlock2.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef SYS_mlock2
LLVM_LIBC_FUNCTION(int, mlock2,
                   (const void *addr, size_t len, unsigned int flags)) {
  auto result = linux_syscalls::mlock2(addr, len, flags);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}
#endif

} // namespace LIBC_NAMESPACE_DECL
