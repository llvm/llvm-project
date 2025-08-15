//===-- Definition of __barrier_type type ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES__BARRIER_TYPE_H
#define LLVM_LIBC_TYPES__BARRIER_TYPE_H

#include "src/__support/threads/CndVar.h"
#include "src/__support/threads/mutex.h"

typedef struct __attribute__((aligned(8 /* alignof (Barrier) */))) {
  unsigned expected;
  unsigned waiting;
  bool blocking;
  char entering[sizeof(LIBC_NAMESPACE::CndVar)];
  char exiting[sizeof(LIBC_NAMESPACE::CndVar)];
  char mutex[sizeof(LIBC_NAMESPACE::Mutex)];
} __barrier_type;

#endif // LLVM_LIBC_TYPES__BARRIER_TYPE_H
