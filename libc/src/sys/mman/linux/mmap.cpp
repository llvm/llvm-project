//===---------- Linux implementation of the POSIX mmap function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mmap.h"
#include "hdr/sys_mman_macros.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, mmap,
                   (void *addr, size_t size, int prot, int flags, int fd,
                    off_t offset)) {
  auto result = linux_syscalls::mmap(addr, size, prot, flags, fd, offset);
  if (!result) {
    libc_errno = result.error();
    return MAP_FAILED;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
