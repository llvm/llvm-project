//===-- Definition of the containerof macro -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_CONTAINEROF_MACRO_H
#define __LLVM_LIBC_MACROS_CONTAINEROF_MACRO_H

#include <llvm-libc-macros/offsetof-macro.h>

#define __containerof(ptr, type, member)                                       \
  ({                                                                           \
    const __typeof(((type *)0)->member) *__ptr = (ptr);                        \
    (type *)(void *)((const char *)__ptr - offsetof(type, member));            \
  })

#endif // __LLVM_LIBC_MACROS_CONTAINEROF_MACRO_H
