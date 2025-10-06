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
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, munmap, (void *addr, size_t size)) {
  auto ret = internal::munmap(addr, size);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (LIBC_UNLIKELY(!ret.has_value())) {
    libc_errno = ret.error();
    return -1;
  }

  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
