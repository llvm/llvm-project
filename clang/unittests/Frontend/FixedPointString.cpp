#include "clang/AST/TypeBase.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using clang::FixedPointValueToString;
using llvm::APSInt;
using llvm::SmallString;

namespace {

TEST(FixedPointString, DifferentTypes) {
  SmallString<64> S;
  FixedPointValueToString(S, APSInt::get(320), 7);
  ASSERT_STREQ(S.c_str(), "2.5");

  S.clear();
  FixedPointValueToString(S, APSInt::get(0), 7);
  ASSERT_STREQ(S.c_str(), "0.0");

  // signed short _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(16, /*Unsigned=*/false), 7);
  ASSERT_STREQ(S.c_str(), "255.9921875");

  // signed _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(32, /*Unsigned=*/false), 15);
  ASSERT_STREQ(S.c_str(), "65535.999969482421875");

  // signed long _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(64, /*Unsigned=*/false), 31);
  ASSERT_STREQ(S.c_str(), "4294967295.9999999995343387126922607421875");

  // unsigned short _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(16, /*Unsigned=*/true), 8);
  ASSERT_STREQ(S.c_str(), "255.99609375");

  // unsigned _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(32, /*Unsigned=*/true), 16);
  ASSERT_STREQ(S.c_str(), "65535.9999847412109375");

  // unsigned long _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(64, /*Unsigned=*/true), 32);
  ASSERT_STREQ(S.c_str(), "4294967295.99999999976716935634613037109375");

  // signed short _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(8, /*Unsigned=*/false), 7);
  ASSERT_STREQ(S.c_str(), "0.9921875");

  // signed _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(16, /*Unsigned=*/false), 15);
  ASSERT_STREQ(S.c_str(), "0.999969482421875");

  // signed long _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(32, /*Unsigned=*/false), 31);
  ASSERT_STREQ(S.c_str(), "0.9999999995343387126922607421875");

  // unsigned short _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(8, /*Unsigned=*/true), 8);
  ASSERT_STREQ(S.c_str(), "0.99609375");

  // unsigned _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(16, /*Unsigned=*/true), 16);
  ASSERT_STREQ(S.c_str(), "0.9999847412109375");

  // unsigned long _Fract
  S.clear();
  FixedPointValueToString(S, APSInt::getMaxValue(32, /*Unsigned=*/true), 32);
  ASSERT_STREQ(S.c_str(), "0.99999999976716935634613037109375");
}

TEST(FixedPointString, Negative) {
  SmallString<64> S;
  FixedPointValueToString(S, APSInt::get(-320), 7);
  ASSERT_STREQ(S.c_str(), "-2.5");

  S.clear();
  FixedPointValueToString(S, APSInt::get(-64), 7);
  ASSERT_STREQ(S.c_str(), "-0.5");

  // signed short _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMinValue(16, /*Unsigned=*/false), 7);
  ASSERT_STREQ(S.c_str(), "-256.0");

  // signed _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMinValue(32, /*Unsigned=*/false), 15);
  ASSERT_STREQ(S.c_str(), "-65536.0");

  // signed long _Accum
  S.clear();
  FixedPointValueToString(S, APSInt::getMinValue(64, /*Unsigned=*/false), 31);
  ASSERT_STREQ(S.c_str(), "-4294967296.0");
}

} // namespace
