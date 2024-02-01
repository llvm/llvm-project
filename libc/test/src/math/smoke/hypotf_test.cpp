//===-- Unittests for hypotf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypotf.h"

using LlvmLibcHypotfTest = HypotTestTemplate<float>;

TEST_F(LlvmLibcHypotfTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypotf);
}
