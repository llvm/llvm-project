//===-- Definition of fd_set type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_FD_SET_H__
#define __LLVM_LIBC_TYPES_FD_SET_H__

#include <llvm-libc-macros/sys-select-macros.h> // FD_SETSIZE

typedef struct {
  __FD_SET_WORD_TYPE __set[__FD_SET_ARRAYSIZE];
} fd_set;

#endif // __LLVM_LIBC_TYPES_FD_SET_H__
