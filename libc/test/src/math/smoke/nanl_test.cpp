//===-- Unittests for nanl ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/sanitizer.h"
#include "src/math/nanl.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
#define SELECT_LONG_DOUBLE(val, _, __) val
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
#define SELECT_LONG_DOUBLE(_, val, __) val
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
#define SELECT_LONG_DOUBLE(_, __, val) val
#else
#error "Unknown long double type"
#endif

class LlvmLibcNanlTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  using StorageType = LIBC_NAMESPACE::fputil::FPBits<long double>::StorageType;

  void run_test(const char *input_str, StorageType bits) {
    long double result = LIBC_NAMESPACE::nanl(input_str);
    auto actual_fp = LIBC_NAMESPACE::fputil::FPBits<long double>(result);
    auto expected_fp = LIBC_NAMESPACE::fputil::FPBits<long double>(bits);
    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
  }
};

TEST_F(LlvmLibcNanlTest, NCharSeq) {
  run_test("",
           SELECT_LONG_DOUBLE(0x7ff8000000000000, (UInt128(0x7fffc00000) << 40),
                              (UInt128(0x7fff800000000000) << 64)));
  run_test("1234", SELECT_LONG_DOUBLE(
                       0x7ff80000000004d2,
                       (UInt128(0x7fffc00000) << 40) + UInt128(0x4d2),
                       (UInt128(0x7fff800000000000) << 64) + UInt128(0x4d2)));
  run_test("0x1234",
           SELECT_LONG_DOUBLE(0x7ff8000000001234,
                              (UInt128(0x7fffc00000) << 40) + UInt128(0x1234),
                              (UInt128(0x7fff800000000000) << 64) +
                                  UInt128(0x1234)));
  run_test("1a",
           SELECT_LONG_DOUBLE(0x7ff8000000000000, (UInt128(0x7fffc00000) << 40),
                              (UInt128(0x7fff800000000000) << 64)));
  run_test("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_",
           SELECT_LONG_DOUBLE(0x7ff8000000000000, (UInt128(0x7fffc00000) << 40),
                              (UInt128(0x7fff800000000000) << 64)));
  run_test("10000000000000000000000000000000000000000000000000",
           SELECT_LONG_DOUBLE(0x7ff8000000000000, (UInt128(0x7fffc00000) << 40),
                              (UInt128(0x7fff800000000000) << 64)));
}

TEST_F(LlvmLibcNanlTest, RandomString) {
  StorageType expected =
      SELECT_LONG_DOUBLE(0x7ff8000000000000, (UInt128(0x7fffc00000) << 40),
                         (UInt128(0x7fff800000000000) << 64));

  run_test(" 1234", expected);
  run_test("-1234", expected);
  run_test("asd&f", expected);
  run_test("123 ", expected);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST_F(LlvmLibcNanlTest, InvalidInput) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::nanl(nullptr); });
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
