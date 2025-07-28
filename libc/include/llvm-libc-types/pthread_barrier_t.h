//===-- Definition of pthread_barrier_t type --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_PTHREAD_BARRIER_T_H
#define LLVM_LIBC_TYPES_PTHREAD_BARRIER_T_H

#include "__barrier_type.h"
typedef __barrier_type pthread_barrier_t;

#endif // LLVM_LIBC_TYPES_PTHREAD_BARRIER_T_H
