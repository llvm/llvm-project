//===-- Definition of struct iovec ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_IOVEC_H__
#define __LLVM_LIBC_TYPES_STRUCT_IOVEC_H__

#include <llvm-libc-types/size_t.h>

struct iovec {
  void *iov_base;
  size_t iov_len;
};

#endif // __LLVM_LIBC_TYPES_STRUCT_IOVEC_H__
