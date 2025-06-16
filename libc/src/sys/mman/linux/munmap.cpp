//===---------- Linux implementation of the POSIX munmap function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/munmap.h"

#include "src/__support/OSUtil/munmap.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, munmap, (void *addr, size_t size)) {
  return internal::munmap(addr, size);
}

} // namespace LIBC_NAMESPACE_DECL
