//===-- Definition of mtx_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_MTX_T_H__
#define __LLVM_LIBC_TYPES_MTX_T_H__

#include <llvm-libc-types/__futex_word.h>

typedef struct {
#ifdef __unix__
  __futex_word __ftxw;
#else
#error "mtx_t type not defined for the target platform."
#endif
  int __mtx_type;
} mtx_t;

#endif // __LLVM_LIBC_TYPES_MTX_T_H__
