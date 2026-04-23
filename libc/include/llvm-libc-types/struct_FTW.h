//===-- Definition of struct FTW ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_FTW_H
#define LLVM_LIBC_TYPES_STRUCT_FTW_H

struct FTW {
  int base;  // Offset of the filename in the pathname.
  int level; // Depth of the file in the tree.
};

#endif // LLVM_LIBC_TYPES_STRUCT_FTW_H
