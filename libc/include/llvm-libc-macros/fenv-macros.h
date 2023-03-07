//===-- Definition of macros from fenv.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_FENV_MACROS_H
#define __LLVM_LIBC_MACROS_FENV_MACROS_H

#define FE_DIVBYZERO 1
#define FE_INEXACT 2
#define FE_INVALID 4
#define FE_OVERFLOW 8
#define FE_UNDERFLOW 16
#define FE_ALL_EXCEPT                                                          \
  (FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW)

#define FE_DOWNWARD 1
#define FE_TONEAREST 2
#define FE_TOWARDZERO 4
#define FE_UPWARD 8

#define FE_DFL_ENV ((fenv_t *)-1)

#endif // __LLVM_LIBC_MACROS_FENV_MACROS_H
