//===-- Definition of struct timespec -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_TIMESPEC_H__
#define __LLVM_LIBC_TYPES_TIMESPEC_H__

#include <llvm-libc-types/time_t.h>

struct timespec {
  time_t tv_sec;
  long tv_nsec;
};

#endif // __LLVM_LIBC_TYPES_TIMESPEC_H__
