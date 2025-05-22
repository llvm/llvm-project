//===-- Definition of cnd_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_CND_T_H
#define LLVM_LIBC_TYPES_CND_T_H

#include "mtx_t.h"

typedef struct {
  void *__qfront;
  void *__qback;
  mtx_t __qmtx;
} cnd_t;

#endif // LLVM_LIBC_TYPES_CND_T_H
