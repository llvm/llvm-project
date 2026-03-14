//===-- Unittests for hypot -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypot.h"

using LlvmLibcHypotTest = HypotTestTemplate<double>;

TEST_F(LlvmLibcHypotTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypot);
}

TEST_F(LlvmLibcHypotTest, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::hypot);
}

TEST_F(LlvmLibcHypotTest, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::hypot);
}
