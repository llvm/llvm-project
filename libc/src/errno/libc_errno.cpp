//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#if LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_THREAD_LOCAL

namespace {
LIBC_THREAD_LOCAL int thread_errno;
}

extern "C" LLVM_LIBC_FUNCTION_ATTR int *__llvm_libc_errno() noexcept {
  return &thread_errno;
}

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_SHARED

namespace {
int shared_errno;
}

extern "C" LLVM_LIBC_FUNCTION_ATTR int *__llvm_libc_errno() noexcept {
  return &shared_errno;
}

#endif

} // namespace LIBC_NAMESPACE_DECL

#if defined(LIBC_COPT_PROVIDE_GNU_ALIASES) &&                                  \
    (LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_THREAD_LOCAL ||                        \
     LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_SHARED)
LLVM_LIBC_ADD_FUNCTION_C_ALIAS(__llvm_libc_errno, __errno_location);
#endif
