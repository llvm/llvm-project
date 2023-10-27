//===-- Definition of struct stat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_STAT_H__
#define __LLVM_LIBC_TYPES_STRUCT_STAT_H__

#include <llvm-libc-types/sa_family_t.h>

struct sockaddr {
  sa_family_t sa_family;
  // sa_data is a variable length array. It is provided with a length of one
  // here as a placeholder.
  char sa_data[1];
};

#endif // __LLVM_LIBC_TYPES_STRUCT_STAT_H__
