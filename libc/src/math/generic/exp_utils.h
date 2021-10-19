//===-- Collection of utils for exp and friends -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_EXP_UTILS_H
#define LLVM_LIBC_SRC_MATH_EXP_UTILS_H

#include <stdint.h>

#define EXP2F_TABLE_BITS 5
#define EXP2F_POLY_ORDER 3
#define N (1 << EXP2F_TABLE_BITS)

namespace __llvm_libc {

struct Exp2fDataTable {
  uint64_t tab[1 << EXP2F_TABLE_BITS];
  double shift_scaled;
  double poly[EXP2F_POLY_ORDER];
  double shift;
  double invln2_scaled;
  double poly_scaled[EXP2F_POLY_ORDER];
};

extern const Exp2fDataTable exp2f_data;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_EXP_UTILS_H
