//===-- Utility class to test different flavors of scalbn -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SCALBN_H
#define LLVM_LIBC_TEST_SRC_MATH_SCALBN_H

#include "LdExpTest.h"
#include "test/UnitTest/Test.h"

#define LIST_SCALBN_TESTS(Name, T, U, func)                                    \
  using LlvmLibc##Name##Test = LdExpTestTemplate<T, U>;                        \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibc##Name##Test, PowersOfTwo) { testPowersOfTwo(&func); }        \
  TEST_F(LlvmLibc##Name##Test, OverFlow) { testOverflow(&func); }              \
  TEST_F(LlvmLibc##Name##Test, UnderflowToZeroOnNormal) {                      \
    testUnderflowToZeroOnNormal(&func);                                        \
  }                                                                            \
  TEST_F(LlvmLibc##Name##Test, UnderflowToZeroOnSubnormal) {                   \
    testUnderflowToZeroOnSubnormal(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibc##Name##Test, NormalOperation) { testNormalOperation(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SCALBN_H
