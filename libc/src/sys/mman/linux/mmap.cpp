//===---------- Linux implementation of the POSIX mmap function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mmap.h"

#include "src/__support/OSUtil/mmap.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {


LLVM_LIBC_FUNCTION(void *, mmap,
                   (void *addr, size_t size, int prot, int flags, int fd,
                    off_t offset)) {
  return internal::mmap(addr, size, prot, flags, fd, offset);
}

} // namespace LIBC_NAMESPACE_DECL
