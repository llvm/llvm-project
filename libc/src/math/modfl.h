//===-- Implementation header for modfl -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_MODFL_H
#define LLVM_LIBC_SRC_MATH_MODFL_H

namespace __llvm_libc {

long double modfl(long double x, long double *iptr);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_MODFL_H
