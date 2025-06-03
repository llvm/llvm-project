//===-- Unittests for wmemcmp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wmemcmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWMemcmpTest, CmpZeroByte) {
  // Comparing zero bytes should result in 0.
  const wchar_t *lhs = L"ab";
  const wchar_t *rhs = L"yz";
  EXPECT_EQ(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 0), 0);
}

TEST(LlvmLibcWMemcmpTest, LhsRhsAreTheSame) {
  // Comparing strings of equal value should result in 0.
  const wchar_t *lhs = L"ab";
  const wchar_t *rhs = L"ab";
  EXPECT_EQ(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcWMemcmpTest, LhsBeforeRhsLexically) {
  // z after b, should result in a value less than 0.
  const wchar_t *lhs = L"ab";
  const wchar_t *rhs = L"az";
  EXPECT_LT(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcWMemcmpTest, LhsAfterRhsLexically) {
  // z after b, should result in a value greater than 0.
  const wchar_t *lhs = L"az";
  const wchar_t *rhs = L"ab";
  EXPECT_GT(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcWMemcmpTest, CompareToEmpty) {
  // lhs is nonempty, should result in a value greater than 0.
  const wchar_t *lhs = L"az";
  const wchar_t *rhs = L"";
  EXPECT_GT(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 1), 0);
}

TEST(LlvmLibcWMemcmpTest, LhsAfterRhsLexicallyLong) {
  // b after a, should result in a value greater than 0.
  const wchar_t *lhs = L"aaaaaaaaaaaaab";
  const wchar_t *rhs = L"aaaaaaaaaaaaaa";
  EXPECT_GT(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 15), 0);
}

TEST(LlvmLibcWMemcmpTest, RhsAfterLhsLexicallyLong) {
  // b after a, should result in a value less than 0.
  const wchar_t *lhs = L"aaaaaaaaaaaaaa";
  const wchar_t *rhs = L"aaaaaaaaaaaaab";
  EXPECT_LT(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 15), 0);
}

TEST(LlvmLibcWMemcmpTest, LhsRhsAreTheSameLong) {
  // Comparing strings of equal value should result in 0.
  const wchar_t *lhs = L"aaaaaaaaaaaaaa";
  const wchar_t *rhs = L"aaaaaaaaaaaaaa";
  EXPECT_EQ(LIBC_NAMESPACE::wmemcmp(lhs, rhs, 15), 0);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWMemcmpTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wmemcmp(L"aaaaaaaaaaaaaa", nullptr, 15); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wmemcmp(nullptr, L"aaaaaaaaaaaaaa", 15); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
