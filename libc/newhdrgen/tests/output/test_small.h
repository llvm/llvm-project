//===-- C standard library header test_small-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SMALL_H
#define LLVM_LIBC_TEST_SMALL_H

#include "__llvm-libc-common.h"
#include "llvm-libc-macros/test_small-macros.h"

#define MACRO_A 1

#define MACRO_B 2

#include <llvm-libc-types/type_a.h>
#include <llvm-libc-types/type_b.h>

enum {
  enum_a = value_1,
  enum_b = value_2,
};

__BEGIN_C_DECLS

#ifdef FUNC_A_16
void func_a()CONST_FUNC_A;
#endif // FUNC_A_16

#ifdef FUNC_B_16
int func_b(int, float)CONST_FUNC_B;
#endif // FUNC_B_16

extern obj object_1;
extern obj object_2;

__END_C_DECLS

#endif // LLVM_LIBC_TEST_SMALL_H
