//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of macros to be used with stdlib functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_STDLIB_MACROS_H
#define LLVM_LIBC_MACROS_STDLIB_MACROS_H

#ifndef NULL
#define __need_NULL
#include <stddef.h>
#endif // NULL

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#ifndef MB_CUR_MAX
// Only the "C" locale is supported, and its multibyte conversion functions
// (mbrtowc, wcrtomb, etc.) use UTF-8, which needs up to 4 bytes per character.
#define MB_CUR_MAX 4
#endif // MB_CUR_MAX

#define RAND_MAX 2147483647

#endif // LLVM_LIBC_MACROS_STDLIB_MACROS_H
