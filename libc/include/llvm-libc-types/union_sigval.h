//===-- Definition of type union sigval -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_UNION_SIGVAL_H__
#define __LLVM_LIBC_TYPES_UNION_SIGVAL_H__

union sigval {
  int sival_int;
  void *sival_ptr;
};

#endif // __LLVM_LIBC_TYPES_UNION_SIGVAL_H__
