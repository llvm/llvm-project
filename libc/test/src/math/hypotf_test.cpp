//===-- Unittests for hypotf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"
#include "hypotf_hard_to_round.h"

#include "src/math/hypotf.h"

using LlvmLibcHypotfTest = HypotTestTemplate<float>;

TEST_F(LlvmLibcHypotfTest, SpecialNumbers) {
  test_special_numbers(&__llvm_libc::hypotf);
}

TEST_F(LlvmLibcHypotfTest, SubnormalRange) {
  test_subnormal_range(&__llvm_libc::hypotf);
}

TEST_F(LlvmLibcHypotfTest, NormalRange) {
  test_normal_range(&__llvm_libc::hypotf);
}

TEST_F(LlvmLibcHypotfTest, TrickyInputs) {
  test_input_list(&__llvm_libc::hypotf, N_HARD_TO_ROUND, HYPOTF_HARD_TO_ROUND);
}
