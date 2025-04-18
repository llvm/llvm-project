//===-- Definition of macros for offsetof ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_OFFSETOF_MACROS_H
#define LLVM_LIBC_HDR_OFFSETOF_MACROS_H

#undef offsetof

// Simplify the inclusion if builtin function is available.
#if __has_builtin(__builtin_offsetof)
#define offsetof(t, d) __builtin_offsetof(t, d)
#else
#define __need_offsetof
#include <stddef.h> // compiler resource header
#endif

#endif // LLVM_LIBC_HDR_OFFSETOF_MACROS_H
