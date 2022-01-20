//===-- Definition of cnd_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_CND_T_H__
#define __LLVM_LIBC_TYPES_CND_T_H__

typedef struct {
  void *__qfront;
  void *__qback;
  struct {
    unsigned char __w[4];
    int __t;
  } __qmtx;
} cnd_t;

#endif // __LLVM_LIBC_TYPES_CND_T_H__
