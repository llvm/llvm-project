//===-- Definition of mtx_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_MTX_T_H__
#define __LLVM_LIBC_TYPES_MTX_T_H__

typedef struct {
#if defined(__x86_64__) || defined(__aarch64__)
  // Futex word should be aligned appropriately to allow target atomic
  // instructions. This declaration mimics the internal setup.
  struct {
    _Alignas(sizeof(unsigned int) > _Alignof(unsigned int)
                 ? sizeof(unsigned int)
                 : _Alignof(unsigned int)) unsigned int __word;
  } __futex_word;
#else
#error "Mutex type mtx_t is not available for the target architecture."
#endif
  int __mtx_type;
} mtx_t;

#endif // __LLVM_LIBC_TYPES_MTX_T_H__
