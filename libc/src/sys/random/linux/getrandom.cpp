//===-- Linux implementation of getrandom ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/random/getrandom.h"

#include "src/__support/OSUtil/linux/getrandom.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, getrandom,
                   (void *buf, size_t buflen, unsigned int flags)) {
  auto rand = internal::getrandom(buf, buflen, flags);
  if (!rand.has_value()) {
    libc_errno = static_cast<int>(rand.error());
    return -1;
  }
  return rand.value();
}

} // namespace LIBC_NAMESPACE_DECL
