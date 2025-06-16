//===-- Linux implementation of getrandom ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/random/getrandom.h"

#include "src/__support/OSUtil/getrandom.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, getrandom,
                   (void *buf, size_t buflen, unsigned int flags)) {
  return internal::getrandom(buf, buflen, flags);
}

} // namespace LIBC_NAMESPACE_DECL
