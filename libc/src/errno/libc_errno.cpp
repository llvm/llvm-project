//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libc_errno.h"
#include "src/__support/CPP/atomic.h"

#define LIBC_ERRNO_MODE_UNDEFINED 1
#define LIBC_ERRNO_MODE_THREAD_LOCAL 2
#define LIBC_ERRNO_MODE_SHARED 3
#define LIBC_ERRNO_MODE_EXTERNAL 4
#define LIBC_ERRNO_MODE_SYSTEM 5

#ifndef LIBC_ERRNO_MODE
#if defined(LIBC_FULL_BUILD) || !defined(LIBC_COPT_PUBLIC_PACKAGING)
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_THREAD_LOCAL
#else
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_SYSTEM
#endif
#endif // LIBC_ERRNO_MODE

#if LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_UNDEFINED &&                            \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_THREAD_LOCAL &&                         \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SHARED &&                               \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_EXTERNAL &&                             \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SYSTEM
#error LIBC_ERRNO_MODE must be one of the following values: \
LIBC_ERRNO_MODE_UNDEFINED, \
LIBC_ERRNO_MODE_THREAD_LOCAL, \
LIBC_ERRNO_MODE_SHARED, \
LIBC_ERRNO_MODE_EXTERNAL, \
LIBC_ERRNO_MODE_SYSTEM
#endif

namespace LIBC_NAMESPACE {

// Define the global `libc_errno` instance.
Errno libc_errno;

#if LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_UNDEFINED

void Errno::operator=(int) {}
Errno::operator int() { return 0; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_THREAD_LOCAL

namespace {
LIBC_THREAD_LOCAL int thread_errno;
}

extern "C" {
int *__llvm_libc_errno() { return &thread_errno; }
}

void Errno::operator=(int a) { __libc_errno = a; }
Errno::operator int() { return __libc_errno; }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_GLOBAL

namespace {
cpp::Atomic<int> global_errno;
}

extern "C" {
int *__llvm_libc_errno() { return &global_errno; }
}

void Errno::operator=(int a) {
  __libc_errno.store(a, cpp::MemoryOrder::RELAXED);
}
Errno::operator int() {
  return __libc_errno.load(cpp::MemoryOrder::RELAXED);
}

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_EXTERNAL

extern "C" {
int *__llvm_libc_errno();
}

void Errno::operator=(int a) { *__llvm_libc_errno() = a; }
Errno::operator int() { return *__llvm_libc_errno(); }

#elif LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_SYSTEM

void Errno::operator=(int a) { errno = a; }
Errno::operator int() { return errno; }

#endif

} // namespace LIBC_NAMESPACE
