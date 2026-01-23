//===-- Unittests for hypotf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypotf16.h"

using LlvmLibcHypotf16Test = HypotTestTemplate<float16>;

TEST_F(LlvmLibcHypotf16Test, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::hypotf16);
}

TEST_F(LlvmLibcHypotf16Test, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::hypotf16);
}
