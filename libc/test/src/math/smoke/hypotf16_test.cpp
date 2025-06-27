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

TEST_F(LlvmLibcHypotf16Test, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypotf16);
}
