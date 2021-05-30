//===-- c_api_test.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime.
//
//===----------------------------------------------------------------------===//

#include "c_api.h"
#include "gtest/gtest.h"

TEST(CAPITest, CWrapperFunctionResultInit) {
  OrcRTCWrapperFunctionResult R;
  OrcRTCWrapperFunctionResultInit(&R);

  EXPECT_EQ(R.Size, 0U);
  EXPECT_EQ(R.Data.ValuePtr, nullptr);

  // Check that this value isn't treated as an out-of-band error.
  EXPECT_EQ(OrcRTCWrapperFunctionResultGetOutOfBandError(&R), nullptr);

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultAllocSmall) {
  constexpr size_t SmallAllocSize = sizeof(const char *);

  OrcRTCWrapperFunctionResult R;
  char *DataPtr = OrcRTCWrapperFunctionResultAllocate(&R, SmallAllocSize);

  for (size_t I = 0; I != SmallAllocSize; ++I)
    DataPtr[I] = 0x55 + I;

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, SmallAllocSize);
  for (size_t I = 0; I != SmallAllocSize; ++I)
    EXPECT_EQ(R.Data.Value[I], (char)(0x55 + I))
        << "Unexpected value at index " << I;

  // Check that this value isn't treated as an out-of-band error.
  EXPECT_EQ(OrcRTCWrapperFunctionResultGetOutOfBandError(&R), nullptr);

  // Check that OrcRTCWrapperFunctionResult(Data|Result|Size) and
  // OrcRTCWrapperFunctionResultGetOutOfBandError behave as expected.
  EXPECT_EQ(OrcRTCWrapperFunctionResultData(&R), R.Data.Value);
  EXPECT_EQ(OrcRTCWrapperFunctionResultSize(&R), SmallAllocSize);
  EXPECT_FALSE(OrcRTCWrapperFunctionResultEmpty(&R));
  EXPECT_EQ(OrcRTCWrapperFunctionResultGetOutOfBandError(&R), nullptr);

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultAllocLarge) {
  constexpr size_t LargeAllocSize = sizeof(const char *) + 1;

  OrcRTCWrapperFunctionResult R;
  char *DataPtr = OrcRTCWrapperFunctionResultAllocate(&R, LargeAllocSize);

  for (size_t I = 0; I != LargeAllocSize; ++I)
    DataPtr[I] = 0x55 + I;

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, LargeAllocSize);
  EXPECT_EQ(R.Data.ValuePtr, DataPtr);
  for (size_t I = 0; I != LargeAllocSize; ++I)
    EXPECT_EQ(R.Data.ValuePtr[I], (char)(0x55 + I))
        << "Unexpected value at index " << I;

  // Check that this value isn't treated as an out-of-band error.
  EXPECT_EQ(OrcRTCWrapperFunctionResultGetOutOfBandError(&R), nullptr);

  // Check that OrcRTCWrapperFunctionResult(Data|Result|Size) and
  // OrcRTCWrapperFunctionResultGetOutOfBandError behave as expected.
  EXPECT_EQ(OrcRTCWrapperFunctionResultData(&R), R.Data.ValuePtr);
  EXPECT_EQ(OrcRTCWrapperFunctionResultSize(&R), LargeAllocSize);
  EXPECT_FALSE(OrcRTCWrapperFunctionResultEmpty(&R));
  EXPECT_EQ(OrcRTCWrapperFunctionResultGetOutOfBandError(&R), nullptr);

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultFromRangeSmall) {
  constexpr size_t SmallAllocSize = sizeof(const char *);

  char Source[SmallAllocSize];
  for (size_t I = 0; I != SmallAllocSize; ++I)
    Source[I] = 0x55 + I;

  OrcRTCWrapperFunctionResult R =
      OrcRTCreateCWrapperFunctionResultFromRange(Source, SmallAllocSize);

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, SmallAllocSize);
  for (size_t I = 0; I != SmallAllocSize; ++I)
    EXPECT_EQ(R.Data.Value[I], (char)(0x55 + I))
        << "Unexpected value at index " << I;

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultFromRangeLarge) {
  constexpr size_t LargeAllocSize = sizeof(const char *) + 1;

  char Source[LargeAllocSize];
  for (size_t I = 0; I != LargeAllocSize; ++I)
    Source[I] = 0x55 + I;

  OrcRTCWrapperFunctionResult R =
      OrcRTCreateCWrapperFunctionResultFromRange(Source, LargeAllocSize);

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, LargeAllocSize);
  for (size_t I = 0; I != LargeAllocSize; ++I)
    EXPECT_EQ(R.Data.ValuePtr[I], (char)(0x55 + I))
        << "Unexpected value at index " << I;

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultFromStringSmall) {
  constexpr size_t SmallAllocSize = sizeof(const char *);

  char Source[SmallAllocSize];
  for (size_t I = 0; I != SmallAllocSize - 1; ++I)
    Source[I] = 'a' + I;
  Source[SmallAllocSize - 1] = '\0';

  OrcRTCWrapperFunctionResult R =
      OrcRTCreateCWrapperFunctionResultFromString(Source);

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, SmallAllocSize);
  for (size_t I = 0; I != SmallAllocSize - 1; ++I)
    EXPECT_EQ(R.Data.Value[I], (char)('a' + I))
        << "Unexpected value at index " << I;
  EXPECT_EQ(R.Data.Value[SmallAllocSize - 1], '\0')
      << "Unexpected value at index " << (SmallAllocSize - 1);

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultFromStringLarge) {
  constexpr size_t LargeAllocSize = sizeof(const char *) + 1;

  char Source[LargeAllocSize];
  for (size_t I = 0; I != LargeAllocSize - 1; ++I)
    Source[I] = 'a' + I;
  Source[LargeAllocSize - 1] = '\0';

  OrcRTCWrapperFunctionResult R =
      OrcRTCreateCWrapperFunctionResultFromString(Source);

  // Check that the inline storage in R.Data.Value contains the expected
  // sequence.
  EXPECT_EQ(R.Size, LargeAllocSize);
  for (size_t I = 0; I != LargeAllocSize - 1; ++I)
    EXPECT_EQ(R.Data.ValuePtr[I], (char)('a' + I))
        << "Unexpected value at index " << I;
  EXPECT_EQ(R.Data.ValuePtr[LargeAllocSize - 1], '\0')
      << "Unexpected value at index " << (LargeAllocSize - 1);

  // Check that we can dispose of the value.
  OrcRTDisposeCWrapperFunctionResult(&R);
}

TEST(CAPITest, CWrapperFunctionResultFromOutOfBandError) {
  constexpr const char *ErrMsg = "test error message";
  OrcRTCWrapperFunctionResult R =
      OrcRTCreateCWrapperFunctionResultFromOutOfBandError(ErrMsg);

#ifndef NDEBUG
  EXPECT_DEATH({ OrcRTCWrapperFunctionResultData(&R); },
               "Cannot get data for out-of-band error value");
  EXPECT_DEATH({ OrcRTCWrapperFunctionResultSize(&R); },
               "Cannot get size for out-of-band error value");
#endif

  EXPECT_FALSE(OrcRTCWrapperFunctionResultEmpty(&R));
  const char *OOBErrMsg = OrcRTCWrapperFunctionResultGetOutOfBandError(&R);
  EXPECT_NE(OOBErrMsg, nullptr);
  EXPECT_NE(OOBErrMsg, ErrMsg);
  EXPECT_TRUE(strcmp(OOBErrMsg, ErrMsg) == 0);

  OrcRTDisposeCWrapperFunctionResult(&R);
}
