//===-- Unittests for nanf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/sanitizer.h"
#include "src/math/nanf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

class LlvmLibcNanf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  using StorageType = LIBC_NAMESPACE::fputil::FPBits<float16>::StorageType;

  void run_test(const char *input_str, StorageType bits) {
    float16 result = LIBC_NAMESPACE::nanf16(input_str);
    auto actual_fp = LIBC_NAMESPACE::fputil::FPBits<float16>(result);
    auto expected_fp = LIBC_NAMESPACE::fputil::FPBits<float16>(bits);
    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
  };
};

TEST_F(LlvmLibcNanf16Test, NCharSeq) {
  run_test("", 0x7e00);
  run_test("123", 0x7e7b);
  run_test("0x123", 0x7f23);
  run_test("1a", 0x7e00);
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           0x7e00);
  run_test("10000000000000000000000000000000000000000000000000", 0x7e00);
}

TEST_F(LlvmLibcNanf16Test, RandomString) {
  run_test(" 1234", 0x7e00);
  run_test("-1234", 0x7e00);
  run_test("asd&f", 0x7e00);
  run_test("123 ", 0x7e00);
}

#ifndef LIBC_HAVE_ADDRESS_SANITIZER
TEST_F(LlvmLibcNanf16Test, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nanf16(nullptr); }, WITH_SIGNAL(SIGSEGV));
}
#endif // LIBC_HAVE_ADDRESS_SANITIZER
