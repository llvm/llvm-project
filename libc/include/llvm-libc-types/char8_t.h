//===-- Definition of char8_t type ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_CHAR8_T_H
#define LLVM_LIBC_TYPES_CHAR8_T_H

// char8_t is only defined as a keyword in C++20, but char16_t and char32_t are
// defined in C++11. Need to guard all of them so our definitions don't conflict
// with the compiler's.
#if !(defined(__cplusplus) && defined(__cpp_char8_t))
typedef unsigned char char8_t;
#endif

#endif // LLVM_LIBC_TYPES_CHAR8_T_H
