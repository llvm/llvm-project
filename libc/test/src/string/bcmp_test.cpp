//===-- Unittests for bcmp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/string/bcmp.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"

namespace __llvm_libc {

TEST(LlvmLibcBcmpTest, CmpZeroByte) {
  const char *lhs = "ab";
  const char *rhs = "bc";
  ASSERT_EQ(__llvm_libc::bcmp(lhs, rhs, 0), 0);
}

TEST(LlvmLibcBcmpTest, LhsRhsAreTheSame) {
  const char *lhs = "ab";
  const char *rhs = "ab";
  ASSERT_EQ(__llvm_libc::bcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcBcmpTest, LhsBeforeRhsLexically) {
  const char *lhs = "ab";
  const char *rhs = "ac";
  ASSERT_NE(__llvm_libc::bcmp(lhs, rhs, 2), 0);
}

TEST(LlvmLibcBcmpTest, LhsAfterRhsLexically) {
  const char *lhs = "ac";
  const char *rhs = "ab";
  ASSERT_NE(__llvm_libc::bcmp(lhs, rhs, 2), 0);
}

// Adapt CheckBcmp signature to op implementation signatures.
template <auto FnImpl>
int CmpAdaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  return FnImpl(p1.begin(), p2.begin(), size);
}

TEST(LlvmLibcBcmpTest, SizeSweep) {
  static constexpr size_t kMaxSize = 1024;
  static constexpr auto Impl = CmpAdaptor<__llvm_libc::bcmp>;
  Buffer Buffer1(kMaxSize);
  Buffer Buffer2(kMaxSize);
  Randomize(Buffer1.span());
  for (size_t size = 0; size < kMaxSize; ++size) {
    auto span1 = Buffer1.span().subspan(0, size);
    auto span2 = Buffer2.span().subspan(0, size);
    const bool OK = CheckBcmp<Impl>(span1, span2, size);
    if (!OK)
      testing::tlog << "Failed at size=" << size << '\n';
    ASSERT_TRUE(OK);
  }
}

} // namespace __llvm_libc
