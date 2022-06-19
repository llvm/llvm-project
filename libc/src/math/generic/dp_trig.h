//===-- Utilities for double precision trigonometric functions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_DP_TRIG_H
#define LLVM_LIBC_SRC_MATH_GENERIC_DP_TRIG_H

namespace __llvm_libc {

double mod_2pi(double);

double mod_pi_over_2(double);

double mod_pi_over_4(double);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GENERIC_DP_TRIG_H
