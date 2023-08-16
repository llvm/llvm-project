#ifndef __LLVM_LIBC_MACROS_LIMITS_MACROS_H
#define __LLVM_LIBC_MACROS_LIMITS_MACROS_H

// Normally compiler headers will be prefered over LLVM-libc headers and
// include_next this header, however, during LLVM-libc build itself the
// LLVM-libc headers are prefered, and to get C numerical limits we need to
// include compiler (freestanding) limits.h. The macro checks are here to avoid
// including limits.h when compiler headers have already been included.

#if !defined _GCC_LIMITS_H_ && !defined __CLANG_LIMITS_H &&                    \
    __has_include_next(<limits.h>)
#include_next <limits.h>
#endif

#ifdef __linux__
#include <linux/limits.h>
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX __LONG_MAX__
#endif

#endif // __LLVM_LIBC_MACROS_LIMITS_MACROS_H
