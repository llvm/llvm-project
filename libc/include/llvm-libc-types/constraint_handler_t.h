//===-- Definition of type constraint_handler_t ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_INCLUDE_LLVM_LIBC_TYPES_CONSTRAINT_HANDLER_T_H
#define LLVM_LIBC_INCLUDE_LLVM_LIBC_TYPES_CONSTRAINT_HANDLER_T_H

#include "errno_t.h"

typedef void (*constraint_handler_t)(const char *__restrict msg,
                                     void *__restrict ptr, errno_t error);

#endif // LLVM_LIBC_INCLUDE_LLVM_LIBC_TYPES_CONSTRAINT_HANDLER_T_H
