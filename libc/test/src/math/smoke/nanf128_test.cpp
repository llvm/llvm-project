//===-- Unittests for nanf128 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/sanitizer.h"
#include "src/__support/uint128.h"
#include "src/math/nanf128.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcNanf128Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  using FPBits128 = LIBC_NAMESPACE::fputil::FPBits<float128>;
  using StorageType = FPBits128::StorageType;

  const UInt128 QUIET_NAN = FPBits128::quiet_nan().uintval();
  const UInt128 ONE = UInt128(1);

  void run_test(const char *input_str, StorageType bits) {
    float128 result = LIBC_NAMESPACE::nanf128(input_str);
    auto actual_fp = FPBits128(result);
    auto expected_fp = FPBits128(bits);
    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
  }
};

TEST_F(LlvmLibcNanf128Test, NCharSeq) {
  run_test("", QUIET_NAN);
  run_test("1234", QUIET_NAN | 1234);
  run_test("0x1234", QUIET_NAN | 0x1234);
  run_test("2417851639229258349412352", QUIET_NAN | (ONE << 81));
  run_test("0x200000000000000000000", QUIET_NAN | (ONE << 81));
  run_test("10384593717069655257060992658440191",
           QUIET_NAN | FPBits128::SIG_MASK);
  run_test("0x1ffffffffffffffffffffffffffff", QUIET_NAN | FPBits128::SIG_MASK);
  run_test("10384593717069655257060992658440192", QUIET_NAN);
  run_test("0x20000000000000000000000000000", QUIET_NAN);
  run_test("1a", QUIET_NAN);
  run_test("10000000000000000000000000000000000000000000000000", QUIET_NAN);
}

TEST_F(LlvmLibcNanf128Test, RandomString) {
  run_test(" 1234", QUIET_NAN);
  run_test("-1234", QUIET_NAN);
  run_test("asd&f", QUIET_NAN);
  run_test("123 ", QUIET_NAN);
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           QUIET_NAN);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST_F(LlvmLibcNanf128Test, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nanf128(nullptr); });
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
