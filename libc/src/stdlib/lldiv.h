//===-- Implementation header for lldiv -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

#ifndef LLVM_LIBC_SRC_STDLIB_LLDIV_H
#define LLVM_LIBC_SRC_STDLIB_LLDIV_H

namespace __llvm_libc {

lldiv_t lldiv(long long x, long long y);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_LDIV_H
