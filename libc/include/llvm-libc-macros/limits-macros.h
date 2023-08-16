#ifndef __LLVM_LIBC_MACROS_LIMITS_MACROS_H
#define __LLVM_LIBC_MACROS_LIMITS_MACROS_H

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
