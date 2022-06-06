//===-- Definition of thrd_t type -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_THREAD_TYPE_H__
#define __LLVM_LIBC_TYPES_THREAD_TYPE_H__

typedef struct {
  void *__attrib;
  void *__platform_attrib;
} __thread_type;

#endif // __LLVM_LIBC_TYPES_THREAD_TYPE_H__
