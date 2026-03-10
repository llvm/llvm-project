//===-- Definition of in_addr type ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_IN_ADDR_H
#define LLVM_LIBC_TYPES_IN_ADDR_H

#include "in_addr_t.h"

typedef struct {
  in_addr_t s_addr;
} in_addr;

#endif // LLVM_LIBC_TYPES_IN_ADDR_H
