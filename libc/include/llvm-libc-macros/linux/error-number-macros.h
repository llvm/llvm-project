#ifndef LLVM_LIBC_MACROS_LINUX_ERROR_NUMBER_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_ERROR_NUMBER_MACROS_H

#if defined(__mips__)
#include "mips/error-number-macros.h"

#elif defined(__sparc__)
#include "sparc/error-number-macros.h"

#else
#ifndef ECANCELED
#define ECANCELED 125
#endif // ECANCELED

#ifndef EOWNERDEAD
#define EOWNERDEAD 130
#endif // EOWNERDEAD

#ifndef ENOTRECOVERABLE
#define ENOTRECOVERABLE 131
#endif // ENOTRECOVERABLE

#ifndef ERFKILL
#define ERFKILL 132
#endif // ERFKILL

#ifndef EHWPOISON
#define EHWPOISON 133
#endif // EHWPOISON
#endif

#endif // LLVM_LIBC_MACROS_LINUX_ERROR_NUMBER_MACROS_H
