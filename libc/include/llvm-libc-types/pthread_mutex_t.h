//===-- Definition of pthread_mutex_t type --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_PTHREAD_MUTEX_T_H
#define __LLVM_LIBC_TYPES_PTHREAD_MUTEX_T_H

#include <llvm-libc-types/__mutex_type.h>

typedef __mutex_type pthread_mutex_t;

#endif // __LLVM_LIBC_TYPES_PTHREAD_MUTEX_T_H
