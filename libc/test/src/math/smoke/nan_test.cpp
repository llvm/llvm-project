//===-- Unittests for nan -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/nan.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <signal.h>

class LlvmLibcNanTest : public LIBC_NAMESPACE::testing::Test {
public:
  using StorageType = LIBC_NAMESPACE::fputil::FPBits<double>::StorageType;

  void run_test(const char *input_str, StorageType bits) {
    double result = LIBC_NAMESPACE::nan(input_str);
    auto actual_fp = LIBC_NAMESPACE::fputil::FPBits<double>(result);
    auto expected_fp = LIBC_NAMESPACE::fputil::FPBits<double>(bits);
    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
  };
};

TEST_F(LlvmLibcNanTest, NCharSeq) {
  run_test("", 0x7ff8000000000000);
  run_test("1234", 0x7ff80000000004d2);
  run_test("0x1234", 0x7ff8000000001234);
  run_test("1a", 0x7ff8000000000000);
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           0x7ff8000000000000);
  run_test("10000000000000000000000000000000000000000000000000",
           0x7ff8000000000000);
}

TEST_F(LlvmLibcNanTest, RandomString) {
  run_test(" 1234", 0x7ff8000000000000);
  run_test("-1234", 0x7ff8000000000000);
  run_test("asd&f", 0x7ff8000000000000);
  run_test("123 ", 0x7ff8000000000000);
}

#ifndef LIBC_HAVE_ADDRESS_SANITIZER
TEST_F(LlvmLibcNanTest, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nan(nullptr); }, WITH_SIGNAL(SIGSEGV));
}
#endif // LIBC_HAVE_ADDRESS_SANITIZER
