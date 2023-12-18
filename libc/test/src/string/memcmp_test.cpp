//===-- Unittests for memcmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcMemcmpTest, CmpZeroByte) {
  const char *lhs = "ab";
  const char *rhs = "yz";
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(lhs, rhs, 0), 0);
}

TEST(LlvmLibcMemcmpTest, LhsRhsAreTheSame) {
  const char *lhs = "ab";
  const char *rhs = "ab";
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcMemcmpTest, LhsBeforeRhsLexically) {
  const char *lhs = "ab";
  const char *rhs = "az";
  EXPECT_LT(LIBC_NAMESPACE::memcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcMemcmpTest, LhsAfterRhsLexically) {
  const char *lhs = "az";
  const char *rhs = "ab";
  EXPECT_GT(LIBC_NAMESPACE::memcmp(lhs, rhs, 2), 0);
}

// Adapt CheckMemcmp signature to memcmp.
static inline int Adaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  return LIBC_NAMESPACE::memcmp(p1.begin(), p2.begin(), size);
}

TEST(LlvmLibcMemcmpTest, SizeSweep) {
  static constexpr size_t kMaxSize = 400;
  Buffer Buffer1(kMaxSize);
  Buffer Buffer2(kMaxSize);
  Randomize(Buffer1.span());
  for (size_t size = 0; size < kMaxSize; ++size) {
    auto span1 = Buffer1.span().subspan(0, size);
    auto span2 = Buffer2.span().subspan(0, size);
    const bool OK = CheckMemcmp<Adaptor>(span1, span2, size);
    if (!OK)
      testing::tlog << "Failed at size=" << size << '\n';
    ASSERT_TRUE(OK);
  }
}

} // namespace LIBC_NAMESPACE
