//===-- Unittests for memcmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "utils/UnitTest/Test.h"

TEST(MemcmpTest, CmpZeroByte) {
  const char *lhs = "ab";
  const char *rhs = "bc";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 0), 0);
}

TEST(MemcmpTest, LhsRhsAreTheSame) {
  const char *lhs = "ab";
  const char *rhs = "ab";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 2), 0);
}

TEST(MemcmpTest, LhsBeforeRhsLexically) {
  const char *lhs = "ab";
  const char *rhs = "ac";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 2), -1);
}

TEST(MemcmpTest, LhsAfterRhsLexically) {
  const char *lhs = "ac";
  const char *rhs = "ab";
  EXPECT_EQ(__llvm_libc::memcmp(lhs, rhs, 2), 1);
}
