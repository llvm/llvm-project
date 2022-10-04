//===-- Definition of __pthread_once_func_t type --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_PTHREAD_ONCE_FUNC_T_H__
#define __LLVM_LIBC_TYPES_PTHREAD_ONCE_FUNC_T_H__

typedef void (*__pthread_once_func_t)(void);

#endif // __LLVM_LIBC_TYPES_PTHREAD_ONCE_FUNC_T_H__
