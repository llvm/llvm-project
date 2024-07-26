//===-- Implementation of libc_errno --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libc_errno.h"
#include "src/__support/macros/config.h"

// libc uses a fallback default value, either system or thread local.
#define LIBC_ERRNO_MODE_DEFAULT 0
// libc never stores a value; `errno` macro uses get link-time failure.
#define LIBC_ERRNO_MODE_UNDEFINED 1
// libc maintains per-thread state (requires C++ `thread_local` support).
#define LIBC_ERRNO_MODE_THREAD_LOCAL 2
// libc maintains shared state used by all threads, contrary to standard C
// semantics unless always single-threaded; nothing prevents data races.
#define LIBC_ERRNO_MODE_SHARED 3
// libc doesn't maintain any internal state, instead the embedder must define
// `int *__llvm_libc_errno(void);` C function.
#define LIBC_ERRNO_MODE_EXTERNAL 4
// libc uses system `<errno.h>` `errno` macro directly in the overlay mode; in
// fullbuild mode, effectively the same as `LIBC_ERRNO_MODE_EXTERNAL`.
#define LIBC_ERRNO_MODE_SYSTEM 5

#if !defined(LIBC_ERRNO_MODE) || LIBC_ERRNO_MODE == LIBC_ERRNO_MODE_DEFAULT
#undef LIBC_ERRNO_MODE
#if defined(LIBC_FULL_BUILD) || !defined(LIBC_COPT_PUBLIC_PACKAGING)
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_THREAD_LOCAL
#else
#define LIBC_ERRNO_MODE LIBC_ERRNO_MODE_SYSTEM
#endif
#endif // LIBC_ERRNO_MODE

#if LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_DEFAULT &&                              \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_UNDEFINED &&                            \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_THREAD_LOCAL &&                         \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SHARED &&                               \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_EXTERNAL &&                             \
    LIBC_ERRNO_MODE != LIBC_ERRNO_MODE_SYSTEM
#error LIBC_ERRNO_MODE must be one of the following values: \
LIBC_ERRNO_MODE_DEFAULT, \
LIBC_ERRNO_MODE_UNDEFINED, \
LIBC_ERRNO_MODE_THREAD_LOCAL, \
LIBC_ERRNO_MODE_SHARED, \
LIBC_ERRNO_MODE_EXTERNAL, \
LIBC_ERRNO_MODE_SYSTEM
#endif

namespace LIBC_NAMESPACE_DECL {

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

} // namespace LIBC_NAMESPACE_DECL
