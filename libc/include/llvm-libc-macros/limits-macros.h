#ifndef __LLVM_LIBC_MACROS_LIMITS_MACROS_H
#define __LLVM_LIBC_MACROS_LIMITS_MACROS_H

#ifdef __linux__
#include <linux/limits.h>
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX __LONG_MAX__
#endif

#endif // __LLVM_LIBC_MACROS_LIMITS_MACROS_H
