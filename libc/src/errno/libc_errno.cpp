//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libc_errno.h"
#include "src/__support/CPP/atomic.h"

#define LIBC_ERRNO_MODE_NONE 0x01
#define LIBC_ERRNO_MODE_INTERNAL 0x02
#define LIBC_ERRNO_MODE_EXTERNAL 0x04
#define LIBC_ERRNO_MODE_THREAD_LOCAL 0x08
#define LIBC_ERRNO_MODE_GLOBAL 0x10
#define LIBC_ERRNO_MODE_LOCATION 0x20

#ifndef LIBC_ERRNO_MODE
#ifndef LIBC_COPT_PUBLIC_PACKAGING
// This mode is for unit testing. We just use our internal errno.
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_INTERNAL
#elif defined(LIBC_FULL_BUILD)
// In full build mode, we provide the errno storage ourselves.
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_THREAD_LOCAL
#else
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_EXTERNAL
#endif
#endif // LIBC_ERRNO_MODE

#if LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_NONE &&                                 \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_INTERNAL &&                             \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_EXTERNAL &&                             \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_THREAD_LOCAL &&                         \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_GLOBAL
#error LIBC_ERRNO_MODE must be one of the following values: \
LIBC_ERRNO_MODE_NONE, \
LIBC_ERRNO_MODE_INTERNAL, \
LIBC_ERRNO_MODE_EXTERNAL, \
LIBC_ERRNO_MODE_THREAD_LOCAL, \
LIBC_ERRNO_MODE_GLOBAL
#endif

namespace LIBC_NAMESPACE {

// Define the global `libc_errno` instance.
Errno libc_errno;

#if LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_NONE

extern "C" {
const int __llvmlibc_errno = 0;
int *__errno_location(void) { return &__llvmlibc_errno; }
}

void Errno::operator=(int) {}
Errno::operator int() { return 0; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_INTERNAL

LIBC_THREAD_LOCAL int __llvmlibc_internal_errno;

extern "C" {
int *__errno_location(void) { return &__llvmlibc_internal_errno; }
}

void Errno::operator=(int a) { __llvmlibc_internal_errno = a; }
Errno::operator int() { return __llvmlibc_internal_errno; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_EXTERNAL

extern "C" {
int *__errno_location(void);
}

void Errno::operator=(int a) { *__errno_location() = a; }
Errno::operator int() { return *__errno_location(); }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_THREAD_LOCAL

extern "C" {
LIBC_THREAD_LOCAL int __llvmlibc_errno;
int *__errno_location(void) { return &__llvmlibc_errno; }
}

void Errno::operator=(int a) { __llvmlibc_errno = a; }
Errno::operator int() { return __llvmlibc_errno; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_GLOBAL

extern "C" {
LIBC_NAMESPACE::cpp::Atomic<int> __llvmlibc_errno;
int *__errno_location(void) { return &__llvmlibc_errno; }
}

void Errno::operator=(int a) {
  __llvmlibc_errno.store(a, cpp::MemoryOrder::RELAXED);
}
Errno::operator int() {
  return __llvmlibc_errno.load(cpp::MemoryOrder::RELAXED);
}

#endif

} // namespace LIBC_NAMESPACE
