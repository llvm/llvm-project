//===-- Unittests tests for hypotbf16 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/hypotbf16.h"

using LlvmLibcHypotbf16Test = HypotTestTemplate<bfloat16>;

TEST_F(LlvmLibcHypotbf16Test, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::hypotbf16);
}

TEST_F(LlvmLibcHypotbf16Test, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::hypotbf16);
}
