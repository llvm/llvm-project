//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#if LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SYSTEM_INLINE

#if LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_UNDEFINED

void Errno::operator=(int) {}
Errno::operator int() { return 0; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_THREAD_LOCAL

namespace {
LIBC_THREAD_LOCAL int thread_errno;
}

extern "C" int *__llvm_libc_errno() noexcept { return &thread_errno; }

void Errno::operator=(int a) { thread_errno = a; }
Errno::operator int() { return thread_errno; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_SHARED

namespace {
int shared_errno;
}

extern "C" int *__llvm_libc_errno() noexcept { return &shared_errno; }

void Errno::operator=(int a) { shared_errno = a; }
Errno::operator int() { return shared_errno; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_EXTERNAL

void Errno::operator=(int a) { *__llvm_libc_errno() = a; }
Errno::operator int() { return *__llvm_libc_errno(); }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_SYSTEM

void Errno::operator=(int a) { errno = a; }
Errno::operator int() { return errno; }

#endif

// Define the global `libc_errno` instance.
Errno libc_errno;

#endif // LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SYSTEM_INLINE

} // namespace LIBC_NAMESPACE_DECL
