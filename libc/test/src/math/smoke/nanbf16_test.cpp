//===-- Unittests for nanbf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/nanbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcNanf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  using StorageType = LIBC_NAMESPACE::fputil::FPBits<bfloat16>::StorageType;

  void run_test(const char *input_str, StorageType bits) {
    bfloat16 result = LIBC_NAMESPACE::nanbf16(input_str);
    auto actual_fp = LIBC_NAMESPACE::fputil::FPBits<bfloat16>(result);
    auto expected_fp = LIBC_NAMESPACE::fputil::FPBits<bfloat16>(bits);
    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
  }
};

TEST_F(LlvmLibcNanf16Test, NCharSeq) {
  run_test("", 0x7fc0);

  // 0x7fc0 + 0x1f (31) = 0x7cdf
  run_test("31", 0x7fdf);

  // 0x7fc0 + 0x15 = 0x7fd5
  run_test("0x15", 0x7fd5);

  run_test("1a", 0x7fc0);
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           0x7fc0);
  run_test("10000000000000000000000000000", 0x7fc0);
}

TEST_F(LlvmLibcNanf16Test, RandomString) {
  run_test(" 1234", 0x7fc0);
  run_test("-1234", 0x7fc0);
  run_test("asd&f", 0x7fc0);
  run_test("123 ", 0x7fc0);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcNanf16Test, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nanbf16(nullptr); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
