//===-- Definition of the type __pthread_tss_dtor_t -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_PTHREAD_TSS_DTOR_T_H__
#define __LLVM_LIBC_TYPES_PTHREAD_TSS_DTOR_T_H__

typedef void (*__pthread_tss_dtor_t)(void *);

#endif // __LLVM_LIBC_TYPES_PTHREAD_TSS_DTOR_T_H__
