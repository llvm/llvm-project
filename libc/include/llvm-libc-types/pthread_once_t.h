//===-- Definition of pthread_once_t type ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H__
#define __LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H__

#include <llvm-libc-types/__futex_word.h>

#ifdef __unix__
typedef __futex_word pthread_once_t;
#else
#error "Once flag type not defined for the target platform."
#endif

#endif // __LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H__
