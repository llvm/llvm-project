//===-- wrappers_c.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_WRAPPERS_C_H_
#define SCUDO_WRAPPERS_C_H_

#include "platform.h"
#include "stats.h"

// Bionic's struct mallinfo consists of size_t (mallinfo(3) uses int).
#if SCUDO_ANDROID
typedef size_t __scudo_mallinfo_data_t;
#else
typedef int __scudo_mallinfo_data_t;
#endif

struct __scudo_mallinfo {
  __scudo_mallinfo_data_t arena;
  __scudo_mallinfo_data_t ordblks;
  __scudo_mallinfo_data_t smblks;
  __scudo_mallinfo_data_t hblks;
  __scudo_mallinfo_data_t hblkhd;
  __scudo_mallinfo_data_t usmblks;
  __scudo_mallinfo_data_t fsmblks;
  __scudo_mallinfo_data_t uordblks;
  __scudo_mallinfo_data_t fordblks;
  __scudo_mallinfo_data_t keepcost;
};

// Android sometimes includes malloc.h no matter what, which yields to
// conflicting return types for mallinfo() if we use our own structure. So if
// struct mallinfo is declared (#define courtesy of malloc.h), use it directly.
#if STRUCT_MALLINFO_DECLARED
#define SCUDO_MALLINFO mallinfo
#else
#define SCUDO_MALLINFO __scudo_mallinfo
#endif

#ifndef M_DECAY_TIME
#define M_DECAY_TIME -100
#endif

#ifndef M_PURGE
#define M_PURGE -101
#endif

#endif // SCUDO_WRAPPERS_C_H_
