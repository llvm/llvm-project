//===-- Standard C header <func_guarding.h> --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LLVM_LIBC_FUNC_GUARDING_H
#define _LLVM_LIBC_FUNC_GUARDING_H

#include "__llvm-libc-common.h"

__BEGIN_C_DECLS

#ifdef LIBC_TYPES_HAS_FLOAT128
void func_all_guarded(int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16
int func_guarded_a(int) __NOEXCEPT;

int func_guarded_b(int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT128
int func_guarded_c(int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

int func_plain(int) __NOEXCEPT;

__END_C_DECLS

#endif // _LLVM_LIBC_FUNC_GUARDING_H
