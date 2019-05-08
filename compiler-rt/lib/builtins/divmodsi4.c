//===-- divmodsi4.c - Implement __divmodsi4
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __divmodsi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

// Returns: a / b, *rem = a % b

COMPILER_RT_ABI si_int __divmodsi4(si_int a, si_int b, si_int *rem) {
  si_int d = __divsi3(a, b);
  *rem = a - (d * b);
  return d;
}
