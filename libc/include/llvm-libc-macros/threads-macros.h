//===-- Definition of threads macros --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_THREADS_MACRO_H
#define LLVM_LIBC_MACROS_THREADS_MACRO_H

#include "__llvm-libc-common.h"
#include "stdint-macros.h"

// LLVM libc extensions
#define __THRD_GET_ID(t)                                                       \
  __LLVM_LIBC_CAST(reinterpret_cast, uintptr_t, (t).__attrib)

#endif // LLVM_LIBC_MACROS_THREADS_MACRO_H
