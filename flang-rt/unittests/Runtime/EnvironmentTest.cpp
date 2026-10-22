//===-- unittests/Runtime/EnvironmentTest.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for parsing of the FORT_CONVERT_UNIT environment variable by
// ParseFortConvertUnit() and the per-unit lookup done by UnitRtConvert().
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/environment.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace Fortran::runtime;

namespace {

class FortConvertUnitTest : public ::testing::Test {
protected:
  void TearDown() override {
    std::free(env.convertUnits);
    env.convertUnits = nullptr;
    env.numConvertUnits = 0;
  }

  ExecutionEnvironment env;
};

} // namespace

TEST_F(FortConvertUnitTest, SingleUnit) {
  ASSERT_TRUE(env.ParseFortConvertUnit("swap:10"));
  EXPECT_EQ(env.conversion, Convert::Unknown);
  ASSERT_EQ(env.numConvertUnits, 1u);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(9), Convert::Unknown);
  EXPECT_EQ(env.UnitRtConvert(11), Convert::Unknown);
}

TEST_F(FortConvertUnitTest, MultipleExceptionsAndUnitLists) {
  // ';'-separated exceptions, with ',' lists and ranges in the first.
  ASSERT_TRUE(env.ParseFortConvertUnit("swap:10,11,15-20;big_endian:30"));
  ASSERT_EQ(env.numConvertUnits, 4u);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(11), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(12), Convert::Unknown);
  EXPECT_EQ(env.UnitRtConvert(15), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(17), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(20), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(21), Convert::Unknown);
  EXPECT_EQ(env.UnitRtConvert(30), Convert::BigEndian);
}

TEST_F(FortConvertUnitTest, EveryMode) {
  ASSERT_TRUE(
      env.ParseFortConvertUnit("native:1;little_endian:2;big_endian:3;swap:4"));
  ASSERT_EQ(env.numConvertUnits, 4u);
  EXPECT_EQ(env.UnitRtConvert(1), Convert::Native);
  EXPECT_EQ(env.UnitRtConvert(2), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(3), Convert::BigEndian);
  EXPECT_EQ(env.UnitRtConvert(4), Convert::Swap);
}

TEST_F(FortConvertUnitTest, ModeIsCaseInsensitive) {
  ASSERT_TRUE(env.ParseFortConvertUnit("SWAP:10;Little_Endian:11"));
  ASSERT_EQ(env.numConvertUnits, 2u);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(11), Convert::LittleEndian);
}

TEST_F(FortConvertUnitTest, ManyTokensTerminate) {
  // Tokens at both levels; each must be consumed exactly once.
  ASSERT_TRUE(env.ParseFortConvertUnit(
      "swap:1,2,3,4,5;native:6,7,8;little_endian:9-12,14;big_endian:20"));
  EXPECT_EQ(env.numConvertUnits, 11u);
  EXPECT_EQ(env.UnitRtConvert(5), Convert::Swap);
  EXPECT_EQ(env.UnitRtConvert(7), Convert::Native);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(13), Convert::Unknown);
  EXPECT_EQ(env.UnitRtConvert(14), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(20), Convert::BigEndian);
}

TEST_F(FortConvertUnitTest, GlobalModeOnly) {
  ASSERT_TRUE(env.ParseFortConvertUnit("swap"));
  EXPECT_EQ(env.conversion, Convert::Swap);
  EXPECT_EQ(env.numConvertUnits, 0u);
  EXPECT_EQ(env.convertUnits, nullptr);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::Unknown);
}

TEST_F(FortConvertUnitTest, GlobalModeThenExceptions) {
  ASSERT_TRUE(env.ParseFortConvertUnit("swap;big_endian:10,12"));
  EXPECT_EQ(env.conversion, Convert::Swap);
  ASSERT_EQ(env.numConvertUnits, 2u);
  EXPECT_EQ(env.UnitRtConvert(10), Convert::BigEndian);
  EXPECT_EQ(env.UnitRtConvert(11), Convert::Unknown);
  EXPECT_EQ(env.UnitRtConvert(12), Convert::BigEndian);
}

TEST_F(FortConvertUnitTest, MissingModeDefaultsToBigEndian) {
  ASSERT_TRUE(env.ParseFortConvertUnit("20-25;little_endian:26-27"));
  ASSERT_EQ(env.numConvertUnits, 2u);
  EXPECT_EQ(env.UnitRtConvert(20), Convert::BigEndian);
  EXPECT_EQ(env.UnitRtConvert(25), Convert::BigEndian);
  EXPECT_EQ(env.UnitRtConvert(26), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(27), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(28), Convert::Unknown);
}

TEST_F(FortConvertUnitTest, LastExceptionWins) {
  ASSERT_TRUE(env.ParseFortConvertUnit("little_endian:10-20;big_endian:19"));
  EXPECT_EQ(env.UnitRtConvert(15), Convert::LittleEndian);
  EXPECT_EQ(env.UnitRtConvert(19), Convert::BigEndian);
  EXPECT_EQ(env.UnitRtConvert(20), Convert::LittleEndian);
}

TEST_F(FortConvertUnitTest, EmptyStringIsNoOp) {
  ASSERT_TRUE(env.ParseFortConvertUnit(""));
  EXPECT_EQ(env.conversion, Convert::Unknown);
  EXPECT_EQ(env.numConvertUnits, 0u);
  EXPECT_EQ(env.convertUnits, nullptr);
}

TEST_F(FortConvertUnitTest, InvalidInputIsRejected) {
  static const char *const invalid[]{
      "bogus", // unknown global mode
      "bogus:10", // unknown mode in an exception
      "swap:5-3", // range with start > end
      "swap:10x", // trailing junk after a unit
      "swap:10-12x", // trailing junk after a range
      "swap:-1", // negative unit
      "swap:abc", // not a number
      "swap:10,11;big_endian:5-3", // failure after valid entries
  };
  for (const char *str : invalid) {
    SCOPED_TRACE(str);
    env.conversion = Convert::Unknown;
    EXPECT_FALSE(env.ParseFortConvertUnit(str));
    // Everything recorded so far must have been backed out, and the freed
    // array must not be left dangling (ExternalFileUnit::CloseAll frees it
    // again).
    EXPECT_EQ(env.numConvertUnits, 0u);
    EXPECT_EQ(env.convertUnits, nullptr);
    EXPECT_EQ(env.conversion, Convert::Unknown);
  }
}
