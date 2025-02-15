//===-- Definition of struct timespec -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_TIMESPEC_H
#define LLVM_LIBC_TYPES_STRUCT_TIMESPEC_H

#include "time_t.h"

#ifndef _STRUCT_TIMESPEC
#define _STRUCT_TIMESPEC 1
struct timespec {
  time_t tv_sec; /* Seconds.  */
  /* TODO: BIG_ENDIAN may require padding. */
  long tv_nsec; /* Nanoseconds.  */
};
#endif // _STRUCT_TIMESPEC

#endif // LLVM_LIBC_TYPES_STRUCT_TIMESPEC_H
