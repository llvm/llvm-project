//===-- Implementation header for ldiv --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_LDIV_H
#define LLVM_LIBC_SRC_STDLIB_LDIV_H

#include <stdlib.h>

namespace __llvm_libc {

ldiv_t ldiv(long x, long y);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_LDIV_H
