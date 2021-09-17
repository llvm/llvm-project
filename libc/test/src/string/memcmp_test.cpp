//===-- Unittests for memcmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcMemcmpTest, CmpZeroByte) {
  const char *lhs = "ab";
  const char *rhs = "yz";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 0), 0);
}

TEST(LlvmLibcMemcmpTest, LhsRhsAreTheSame) {
  const char *lhs = "ab";
  const char *rhs = "ab";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcMemcmpTest, LhsBeforeRhsLexically) {
  const char *lhs = "ab";
  const char *rhs = "az";
  EXPECT_LT(__llvm_libc::memcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcMemcmpTest, LhsAfterRhsLexically) {
  const char *lhs = "az";
  const char *rhs = "ab";
  EXPECT_GT(__llvm_libc::memcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcMemcmpTest, Sweep) {
  static constexpr size_t kMaxSize = 1024;
  char lhs[kMaxSize];
  char rhs[kMaxSize];

  const auto reset = [](char *const ptr) {
    for (size_t i = 0; i < kMaxSize; ++i)
      ptr[i] = 'a';
  };

  reset(lhs);
  reset(rhs);
  for (size_t i = 0; i < kMaxSize; ++i)
    EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, i), 0);

  reset(lhs);
  reset(rhs);
  for (size_t i = 0; i < kMaxSize; ++i) {
    rhs[i] = 'z';
    EXPECT_LT(__llvm_libc::memcmp(lhs, rhs, kMaxSize), 0);
    rhs[i] = 'a';
  }
}
