//===---------- Linux implementation of the Linux pkey_mprotect function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/pkey_mprotect.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/mprotect.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/pkey_mprotect.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pkey_mprotect,
                   (void *addr, size_t len, int prot, int pkey)) {
  ErrorOr<int> ret(0);
  if (pkey == -1) {
    ret = linux_syscalls::mprotect(addr, len, prot);
  } else {
    ret = linux_syscalls::pkey_mprotect(addr, len, prot, pkey);
  }

  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
