//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libc_errno.h"

#ifdef LIBC_TARGET_ARCH_IS_GPU
// LIBC_THREAD_LOCAL on GPU currently does nothing.  So essentially this is just
// a global errno for gpu to use for now.
extern "C" {
LIBC_THREAD_LOCAL int __llvmlibc_gpu_errno;
}

void LIBC_NAMESPACE::Errno::operator=(int a) { __llvmlibc_gpu_errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return __llvmlibc_gpu_errno; }

#elif !defined(LIBC_COPT_PUBLIC_PACKAGING)
// This mode is for unit testing.  We just use our internal errno.
LIBC_THREAD_LOCAL int __llvmlibc_internal_errno;

void LIBC_NAMESPACE::Errno::operator=(int a) { __llvmlibc_internal_errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return __llvmlibc_internal_errno; }

#elif defined(LIBC_FULL_BUILD)
// This mode is for public libc archive, hermetic, and integration tests.
// In full build mode, we provide the errno storage ourselves.
extern "C" {
LIBC_THREAD_LOCAL int __llvmlibc_errno;
}

void LIBC_NAMESPACE::Errno::operator=(int a) { __llvmlibc_errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return __llvmlibc_errno; }

#else
void LIBC_NAMESPACE::Errno::operator=(int a) { errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return errno; }

#endif // LIBC_FULL_BUILD

namespace LIBC_NAMESPACE {
// Define the global `libc_errno` instance.
Errno libc_errno;
} // namespace LIBC_NAMESPACE
