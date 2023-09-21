//===-- Implementation header for longjmp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SETJMP_LONGJMP_H
#define LLVM_LIBC_SRC_SETJMP_LONGJMP_H

#include <setjmp.h>

namespace __llvm_libc {

void longjmp(__jmp_buf *buf, int val);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SETJMP_LONGJMP_H
