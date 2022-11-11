//===-- Definition of struct timeval -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_TIMEVAL_H__
#define __LLVM_LIBC_TYPES_TIMEVAL_H__

#include <llvm-libc-types/suseconds_t.h>
#include <llvm-libc-types/time_t.h>

struct timeval {
  time_t tv_sec;       // Seconds
  suseconds_t tv_usec; // Micro seconds
};

#endif // __LLVM_LIBC_TYPES_TIMEVAL_H__
