//===-- Definition of char16_t type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_CHAR16_T_H
#define LLVM_LIBC_TYPES_CHAR16_T_H

#if !(defined(__cplusplus) && defined(__cpp_unicode_characters))
#include "../llvm-libc-macros/stdint-macros.h"
typedef uint_least16_t char16_t;
#endif

#endif // LLVM_LIBC_TYPES_CHAR16_T_H
