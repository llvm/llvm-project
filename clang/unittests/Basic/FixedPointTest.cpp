//===- unittests/Basic/FixedPointTest.cpp -- fixed point number tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FixedPoint.h"
#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"

using clang::APFixedPoint;
using clang::FixedPointSemantics;
using llvm::APInt;
using llvm::APSInt;

namespace {

FixedPointSemantics Saturated(FixedPointSemantics Sema) {
  Sema.setSaturated(true);
  return Sema;
}

FixedPointSemantics getSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/7, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/15, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getLAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/31, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/7, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/15, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getLFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/31, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/8, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/16, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getULAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/32, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/8, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/16, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getULFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/32, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getPadUSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/7, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/15, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadULAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/31, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/7, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/15, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadULFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/31, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

void CheckUnpaddedMax(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMax(Sema).getValue(),
            APSInt::getMaxValue(Sema.getWidth(), !Sema.isSigned()));
}

void CheckPaddedMax(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMax(Sema).getValue(),
            APSInt::getMaxValue(Sema.getWidth(), !Sema.isSigned()) >> 1);
}

void CheckMin(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMin(Sema).getValue(),
            APSInt::getMinValue(Sema.getWidth(), !Sema.isSigned()));
}

TEST(FixedPointTest, getMax) {
  CheckUnpaddedMax(getSAccumSema());
  CheckUnpaddedMax(getAccumSema());
  CheckUnpaddedMax(getLAccumSema());
  CheckUnpaddedMax(getUSAccumSema());
  CheckUnpaddedMax(getUAccumSema());
  CheckUnpaddedMax(getULAccumSema());
  CheckUnpaddedMax(getSFractSema());
  CheckUnpaddedMax(getFractSema());
  CheckUnpaddedMax(getLFractSema());
  CheckUnpaddedMax(getUSFractSema());
  CheckUnpaddedMax(getUFractSema());
  CheckUnpaddedMax(getULFractSema());

  CheckPaddedMax(getPadUSAccumSema());
  CheckPaddedMax(getPadUAccumSema());
  CheckPaddedMax(getPadULAccumSema());
  CheckPaddedMax(getPadUSFractSema());
  CheckPaddedMax(getPadUFractSema());
  CheckPaddedMax(getPadULFractSema());
}

TEST(FixedPointTest, getMin) {
  CheckMin(getSAccumSema());
  CheckMin(getAccumSema());
  CheckMin(getLAccumSema());
  CheckMin(getUSAccumSema());
  CheckMin(getUAccumSema());
  CheckMin(getULAccumSema());
  CheckMin(getSFractSema());
  CheckMin(getFractSema());
  CheckMin(getLFractSema());
  CheckMin(getUSFractSema());
  CheckMin(getUFractSema());
  CheckMin(getULFractSema());

  CheckMin(getPadUSAccumSema());
  CheckMin(getPadUAccumSema());
  CheckMin(getPadULAccumSema());
  CheckMin(getPadUSFractSema());
  CheckMin(getPadUFractSema());
  CheckMin(getPadULFractSema());
}

void CheckIntPart(const FixedPointSemantics &Sema, int64_t IntPart) {
  unsigned Scale = Sema.getScale();

  // Value with a fraction
  APFixedPoint ValWithFract(APInt(Sema.getWidth(),
                                  (IntPart << Scale) + (1ULL << (Scale - 1)),
                                  Sema.isSigned()),
                            Sema);
  ASSERT_EQ(ValWithFract.getIntPart(), IntPart);

  // Just fraction
  APFixedPoint JustFract(
      APInt(Sema.getWidth(), (1ULL << (Scale - 1)), Sema.isSigned()), Sema);
  ASSERT_EQ(JustFract.getIntPart(), 0);

  // Whole number
  APFixedPoint WholeNum(
      APInt(Sema.getWidth(), (IntPart << Scale), Sema.isSigned()), Sema);
  ASSERT_EQ(WholeNum.getIntPart(), IntPart);

  // Negative
  if (Sema.isSigned()) {
    APFixedPoint Negative(
        APInt(Sema.getWidth(), (IntPart << Scale), Sema.isSigned()), Sema);
    ASSERT_EQ(Negative.getIntPart(), IntPart);
  }
}

void CheckIntPartMin(const FixedPointSemantics &Sema, int64_t Expected) {
  ASSERT_EQ(APFixedPoint::getMin(Sema).getIntPart(), Expected);
}

void CheckIntPartMax(const FixedPointSemantics &Sema, uint64_t Expected) {
  ASSERT_EQ(APFixedPoint::getMax(Sema).getIntPart(), Expected);
}

TEST(FixedPoint, getIntPart) {
  // Normal values
  CheckIntPart(getSAccumSema(), 2);
  CheckIntPart(getAccumSema(), 2);
  CheckIntPart(getLAccumSema(), 2);
  CheckIntPart(getUSAccumSema(), 2);
  CheckIntPart(getUAccumSema(), 2);
  CheckIntPart(getULAccumSema(), 2);

  // Zero
  CheckIntPart(getSAccumSema(), 0);
  CheckIntPart(getAccumSema(), 0);
  CheckIntPart(getLAccumSema(), 0);
  CheckIntPart(getUSAccumSema(), 0);
  CheckIntPart(getUAccumSema(), 0);
  CheckIntPart(getULAccumSema(), 0);

  CheckIntPart(getSFractSema(), 0);
  CheckIntPart(getFractSema(), 0);
  CheckIntPart(getLFractSema(), 0);
  CheckIntPart(getUSFractSema(), 0);
  CheckIntPart(getUFractSema(), 0);
  CheckIntPart(getULFractSema(), 0);

  // Min
  CheckIntPartMin(getSAccumSema(), -256);
  CheckIntPartMin(getAccumSema(), -65536);
  CheckIntPartMin(getLAccumSema(), -4294967296);

  CheckIntPartMin(getSFractSema(), -1);
  CheckIntPartMin(getFractSema(), -1);
  CheckIntPartMin(getLFractSema(), -1);

  // Max
  CheckIntPartMax(getSAccumSema(), 255);
  CheckIntPartMax(getAccumSema(), 65535);
  CheckIntPartMax(getLAccumSema(), 4294967295);
  CheckIntPartMax(getUSAccumSema(), 255);
  CheckIntPartMax(getUAccumSema(), 65535);
  CheckIntPartMax(getULAccumSema(), 4294967295);

  CheckIntPartMax(getSFractSema(), 0);
  CheckIntPartMax(getFractSema(), 0);
  CheckIntPartMax(getLFractSema(), 0);
  CheckIntPartMax(getUSFractSema(), 0);
  CheckIntPartMax(getUFractSema(), 0);
  CheckIntPartMax(getULFractSema(), 0);

  // Padded
  // Normal Values
  CheckIntPart(getPadUSAccumSema(), 2);
  CheckIntPart(getPadUAccumSema(), 2);
  CheckIntPart(getPadULAccumSema(), 2);

  // Zero
  CheckIntPart(getPadUSAccumSema(), 0);
  CheckIntPart(getPadUAccumSema(), 0);
  CheckIntPart(getPadULAccumSema(), 0);

  CheckIntPart(getPadUSFractSema(), 0);
  CheckIntPart(getPadUFractSema(), 0);
  CheckIntPart(getPadULFractSema(), 0);

  // Max
  CheckIntPartMax(getPadUSAccumSema(), 255);
  CheckIntPartMax(getPadUAccumSema(), 65535);
  CheckIntPartMax(getPadULAccumSema(), 4294967295);

  CheckIntPartMax(getPadUSFractSema(), 0);
  CheckIntPartMax(getPadUFractSema(), 0);
  CheckIntPartMax(getPadULFractSema(), 0);
}

TEST(FixedPoint, compare) {
  // Equality
  // With fractional part (2.5)
  // Across sizes
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(81920, getAccumSema()));
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(5368709120, getLAccumSema()));
  ASSERT_EQ(APFixedPoint(0, getSAccumSema()), APFixedPoint(0, getLAccumSema()));

  // Across types (0.5)
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(64, getSFractSema()));
  ASSERT_EQ(APFixedPoint(16384, getAccumSema()),
            APFixedPoint(16384, getFractSema()));
  ASSERT_EQ(APFixedPoint(1073741824, getLAccumSema()),
            APFixedPoint(1073741824, getLFractSema()));

  // Across widths and types (0.5)
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(16384, getFractSema()));
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(1073741824, getLFractSema()));

  // Across saturation
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(81920, Saturated(getAccumSema())));

  // Across signs
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(640, getUSAccumSema()));
  ASSERT_EQ(APFixedPoint(-320, getSAccumSema()),
            APFixedPoint(-81920, getAccumSema()));

  // Across padding
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(320, getPadUSAccumSema()));
  ASSERT_EQ(APFixedPoint(640, getUSAccumSema()),
            APFixedPoint(320, getPadUSAccumSema()));

  // Less than
  ASSERT_LT(APFixedPoint(-1, getSAccumSema()), APFixedPoint(0, getAccumSema()));
  ASSERT_LT(APFixedPoint(-1, getSAccumSema()),
            APFixedPoint(0, getUAccumSema()));
  ASSERT_LT(APFixedPoint(0, getSAccumSema()), APFixedPoint(1, getAccumSema()));
  ASSERT_LT(APFixedPoint(0, getSAccumSema()), APFixedPoint(1, getUAccumSema()));
  ASSERT_LT(APFixedPoint(0, getUSAccumSema()), APFixedPoint(1, getAccumSema()));
  ASSERT_LT(APFixedPoint(0, getUSAccumSema()),
            APFixedPoint(1, getUAccumSema()));

  // Greater than
  ASSERT_GT(APFixedPoint(0, getAccumSema()), APFixedPoint(-1, getSAccumSema()));
  ASSERT_GT(APFixedPoint(0, getUAccumSema()),
            APFixedPoint(-1, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getAccumSema()), APFixedPoint(0, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getUAccumSema()), APFixedPoint(0, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getAccumSema()), APFixedPoint(0, getUSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getUAccumSema()),
            APFixedPoint(0, getUSAccumSema()));
}

// Check that a fixed point value in one sema is the same in another sema
void CheckUnsaturatedConversion(FixedPointSemantics Src,
                                FixedPointSemantics Dst, int64_t TestVal) {
  int64_t ScaledVal = TestVal;
  bool IsNegative = ScaledVal < 0;
  if (IsNegative)
    ScaledVal = -ScaledVal;

  if (Dst.getScale() > Src.getScale()) {
    ScaledVal <<= (Dst.getScale() - Src.getScale());
  } else {
    ScaledVal >>= (Src.getScale() - Dst.getScale());
  }

  if (IsNegative)
    ScaledVal = -ScaledVal;

  APFixedPoint Fixed(TestVal, Src);
  APFixedPoint Expected(ScaledVal, Dst);
  ASSERT_EQ(Fixed.convert(Dst), Expected);
}

// Check the value in a given fixed point sema overflows to the saturated min
// for another sema
void CheckSaturatedConversionMin(FixedPointSemantics Src,
                                 FixedPointSemantics Dst, int64_t TestVal) {
  APFixedPoint Fixed(TestVal, Src);
  ASSERT_EQ(Fixed.convert(Dst), APFixedPoint::getMin(Dst));
}

// Check the value in a given fixed point sema overflows to the saturated max
// for another sema
void CheckSaturatedConversionMax(FixedPointSemantics Src,
                                 FixedPointSemantics Dst, int64_t TestVal) {
  APFixedPoint Fixed(TestVal, Src);
  ASSERT_EQ(Fixed.convert(Dst), APFixedPoint::getMax(Dst));
}

// Check one signed _Accum sema converted to other sema for different values.
void CheckSignedAccumConversionsAgainstOthers(FixedPointSemantics Src,
                                              int64_t OneVal) {
  int64_t NormalVal = (OneVal * 2) + (OneVal / 2); // 2.5
  int64_t HalfVal = (OneVal / 2);                  // 0.5

  // +Accums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getULAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), NormalVal);

  // -Accums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), -NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), -NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getUSAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getUAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getULAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUSAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadULAccumSema()), -NormalVal);

  // +Accums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getULFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), HalfVal);

  // -Accums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), -HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), -HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getUSFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getUFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getULFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUSFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadULFractSema()), -HalfVal);

  // 0 to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getLAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getUAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getULAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), 0);

  // 0 to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getFractSema(), 0);
  CheckUnsaturatedConversion(Src, getLFractSema(), 0);
  CheckUnsaturatedConversion(Src, getUSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getUFractSema(), 0);
  CheckUnsaturatedConversion(Src, getULFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), 0);
}

// Check one unsigned _Accum sema converted to other sema for different
// values.
void CheckUnsignedAccumConversionsAgainstOthers(FixedPointSemantics Src,
                                                int64_t OneVal) {
  int64_t NormalVal = (OneVal * 2) + (OneVal / 2); // 2.5
  int64_t HalfVal = (OneVal / 2);                  // 0.5

  // +UAccums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getULAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), NormalVal);

  // +UAccums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getULFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), HalfVal);
}

TEST(FixedPoint, AccumConversions) {
  // Normal conversions
  CheckSignedAccumConversionsAgainstOthers(getSAccumSema(), 128);
  CheckUnsignedAccumConversionsAgainstOthers(getUSAccumSema(), 256);
  CheckSignedAccumConversionsAgainstOthers(getAccumSema(), 32768);
  CheckUnsignedAccumConversionsAgainstOthers(getUAccumSema(), 65536);
  CheckSignedAccumConversionsAgainstOthers(getLAccumSema(), 2147483648);
  CheckUnsignedAccumConversionsAgainstOthers(getULAccumSema(), 4294967296);

  CheckUnsignedAccumConversionsAgainstOthers(getPadUSAccumSema(), 128);
  CheckUnsignedAccumConversionsAgainstOthers(getPadUAccumSema(), 32768);
  CheckUnsignedAccumConversionsAgainstOthers(getPadULAccumSema(), 2147483648);
}

TEST(FixedPoint, AccumConversionOverflow) {
  // To SAccum max limit (65536)
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getPadUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getAccumSema()),
                              281474976710656);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getUAccumSema()),
                              281474976710656);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getPadUAccumSema()),
                              281474976710656);

  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getPadULAccumSema(),
                              Saturated(getPadUAccumSema()), 140737488355328);

  // To SAccum min limit (-65536)
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getAccumSema()),
                              -140737488355328);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getUAccumSema()),
                              -140737488355328);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getPadUAccumSema()),
                              -140737488355328);
}

TEST(FixedPoint, SAccumConversionOverflow) {
  // To SAccum max limit (256)
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getPadUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getUSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getPadUSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getPadUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getSAccumSema()),
                              1099511627776);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getUSAccumSema()),
                              1099511627776);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getPadUSAccumSema()),
                              1099511627776);

  CheckSaturatedConversionMax(getPadUAccumSema(), Saturated(getSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getPadUAccumSema(), Saturated(getUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getPadUAccumSema(),
                              Saturated(getPadUSAccumSema()), 8388608);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getPadULAccumSema(),
                              Saturated(getPadUSAccumSema()), 549755813888);

  // To SAccum min limit (-256)
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getUSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getPadUSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getSAccumSema()),
                              -549755813888);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getUSAccumSema()),
                              -549755813888);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getPadUSAccumSema()),
                              -549755813888);
}

TEST(FixedPoint, GetValueSignAfterConversion) {
  APFixedPoint Fixed(255 << 7, getSAccumSema());
  ASSERT_TRUE(Fixed.getValue().isSigned());
  APFixedPoint UFixed = Fixed.convert(getUSAccumSema());
  ASSERT_TRUE(UFixed.getValue().isUnsigned());
  ASSERT_EQ(UFixed.getValue(), APSInt::getUnsigned(255 << 8).extOrTrunc(16));
}

TEST(FixedPoint, ModularWrapAround) {
  // Positive to negative
  APFixedPoint Val = APFixedPoint(1ULL << 7, getSAccumSema());
  ASSERT_EQ(Val.convert(getLFractSema()).getValue(), -(1ULL << 31));

  Val = APFixedPoint(1ULL << 23, getAccumSema());
  ASSERT_EQ(Val.convert(getSAccumSema()).getValue(), -(1ULL << 15));

  Val = APFixedPoint(1ULL << 47, getLAccumSema());
  ASSERT_EQ(Val.convert(getAccumSema()).getValue(), -(1ULL << 31));

  // Negative to positive
  Val = APFixedPoint(/*-1.5*/ -192, getSAccumSema());
  ASSERT_EQ(Val.convert(getLFractSema()).getValue(), 1ULL << 30);

  Val = APFixedPoint(-(257 << 15), getAccumSema());
  ASSERT_EQ(Val.convert(getSAccumSema()).getValue(), 255 << 7);

  Val = APFixedPoint(-(65537ULL << 31), getLAccumSema());
  ASSERT_EQ(Val.convert(getAccumSema()).getValue(), 65535 << 15);

  // Signed to unsigned
  Val = APFixedPoint(-(1 << 7), getSAccumSema());
  ASSERT_EQ(Val.convert(getUSAccumSema()).getValue(), 255 << 8);

  Val = APFixedPoint(-(1 << 15), getAccumSema());
  ASSERT_EQ(Val.convert(getUAccumSema()).getValue(), 65535ULL << 16);

  Val = APFixedPoint(-(1ULL << 31), getLAccumSema());
  ASSERT_EQ(Val.convert(getULAccumSema()).getValue().getZExtValue(),
            4294967295ULL << 32);
}

} // namespace
