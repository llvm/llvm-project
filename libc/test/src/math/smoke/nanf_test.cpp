//===-- Unittests for nanf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/nanf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <signal.h>

class LlvmLibcNanfTest : public LIBC_NAMESPACE::testing::Test {
public:
  using StorageType = LIBC_NAMESPACE::fputil::FPBits<float>::StorageType;

  void run_test(const char *input_str, StorageType bits) {
    float result = LIBC_NAMESPACE::nanf(input_str);
    auto actual_fp = LIBC_NAMESPACE::fputil::FPBits<float>(result);
    auto expected_fp = LIBC_NAMESPACE::fputil::FPBits<float>(bits);
    EXPECT_EQ(actual_fp.bits, expected_fp.bits);
  };
};

TEST_F(LlvmLibcNanfTest, NCharSeq) {
  run_test("", 0x7fc00000);
  run_test("1234", 0x7fc004d2);
  run_test("0x1234", 0x7fc01234);
  run_test("1a", 0x7fc00000);
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           0x7fc00000);
  run_test("10000000000000000000000000000000000000000000000000", 0x7fc00000);
}

TEST_F(LlvmLibcNanfTest, RandomString) {
  run_test(" 1234", 0x7fc00000);
  run_test("-1234", 0x7fc00000);
  run_test("asd&f", 0x7fc00000);
  run_test("123 ", 0x7fc00000);
}

#ifndef LIBC_HAVE_ADDRESS_SANITIZER
TEST_F(LlvmLibcNanfTest, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nanf(nullptr); }, WITH_SIGNAL(SIGSEGV));
}
#endif // LIBC_HAVE_ADDRESS_SANITIZER
