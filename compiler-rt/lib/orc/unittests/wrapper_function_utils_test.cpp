//===-- wrapper_function_utils_test.cpp -----------------------------------===//
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

#include "wrapper_function_utils.h"
#include "gtest/gtest.h"

#include <stdio.h>

using namespace __orc_rt;

namespace {
constexpr const char *TestString = "test string";
} // end anonymous namespace

TEST(WrapperFunctionUtilsTest, DefaultWrapperFunctionResult) {
  WrapperFunctionResult R;
  EXPECT_TRUE(R.empty());
  EXPECT_EQ(R.size(), 0U);
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromCStruct) {
  __orc_rt_CWrapperFunctionResult CR =
      __orc_rt_CreateCWrapperFunctionResultFromString(TestString);
  WrapperFunctionResult R(CR);
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromRange) {
  auto R = WrapperFunctionResult::copyFrom(TestString, strlen(TestString) + 1);
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromCString) {
  auto R = WrapperFunctionResult::copyFrom(TestString);
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromStdString) {
  auto R = WrapperFunctionResult::copyFrom(std::string(TestString));
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromOutOfBandError) {
  auto R = WrapperFunctionResult::createOutOfBandError(TestString);
  EXPECT_FALSE(R.empty());
  EXPECT_TRUE(strcmp(R.getOutOfBandError(), TestString) == 0);
}

TEST(WrapperFunctionUtilsTest, SPSOutputBuffer) {
  constexpr unsigned NumBytes = 8;
  char Buffer[NumBytes];
  char Zero = 0;
  SPSOutputBuffer OB(Buffer, NumBytes);

  // Expect that we can write NumBytes of content.
  for (unsigned I = 0; I != NumBytes; ++I) {
    char C = I;
    EXPECT_TRUE(OB.write(&C, 1));
  }

  // Expect an error when we attempt to write an extra byte.
  EXPECT_FALSE(OB.write(&Zero, 1));

  // Check that the buffer contains the expected content.
  for (unsigned I = 0; I != NumBytes; ++I)
    EXPECT_EQ(Buffer[I], (char)I);
}

TEST(WrapperFunctionUtilsTest, SPSInputBuffer) {
  char Buffer[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
  SPSInputBuffer IB(Buffer, sizeof(Buffer));

  char C;
  for (unsigned I = 0; I != sizeof(Buffer); ++I) {
    EXPECT_TRUE(IB.read(&C, 1));
    EXPECT_EQ(C, (char)I);
  }

  EXPECT_FALSE(IB.read(&C, 1));
}

template <typename SPSTagT, typename T>
static void blobSerializationRoundTrip(const T &Value) {
  using BST = SPSSerializationTraits<SPSTagT, T>;

  size_t Size = BST::size(Value);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BST::serialize(OB, Value));

  SPSInputBuffer IB(Buffer.get(), Size);

  T DSValue;
  EXPECT_TRUE(BST::deserialize(IB, DSValue));

  EXPECT_EQ(Value, DSValue)
      << "Incorrect value after serialization/deserialization round-trip";
}

template <typename T> static void testFixedIntegralTypeSerialization() {
  blobSerializationRoundTrip<T, T>(0);
  blobSerializationRoundTrip<T, T>(static_cast<T>(1));
  if (std::is_signed<T>::value) {
    blobSerializationRoundTrip<T, T>(static_cast<T>(-1));
    blobSerializationRoundTrip<T, T>(std::numeric_limits<T>::min());
  }
  blobSerializationRoundTrip<T, T>(std::numeric_limits<T>::max());
}

TEST(WrapperFunctionUtilsTest, BoolSerialization) {
  blobSerializationRoundTrip<bool, bool>(true);
  blobSerializationRoundTrip<bool, bool>(false);
}

TEST(WrapperFunctionUtilsTest, CharSerialization) {
  blobSerializationRoundTrip<char, char>((char)0x00);
  blobSerializationRoundTrip<char, char>((char)0xAA);
  blobSerializationRoundTrip<char, char>((char)0xFF);
}

TEST(WrapperFunctionUtilsTest, Int8Serialization) {
  testFixedIntegralTypeSerialization<int8_t>();
}

TEST(WrapperFunctionUtilsTest, UInt8Serialization) {
  testFixedIntegralTypeSerialization<uint8_t>();
}

TEST(WrapperFunctionUtilsTest, Int16Serialization) {
  testFixedIntegralTypeSerialization<int16_t>();
}

TEST(WrapperFunctionUtilsTest, UInt16Serialization) {
  testFixedIntegralTypeSerialization<uint16_t>();
}

TEST(WrapperFunctionUtilsTest, Int32Serialization) {
  testFixedIntegralTypeSerialization<int32_t>();
}

TEST(WrapperFunctionUtilsTest, UInt32Serialization) {
  testFixedIntegralTypeSerialization<uint32_t>();
}

TEST(WrapperFunctionUtilsTest, Int64Serialization) {
  testFixedIntegralTypeSerialization<int64_t>();
}

TEST(WrapperFunctionUtilsTest, UInt64Serialization) {
  testFixedIntegralTypeSerialization<uint64_t>();
}

TEST(WrapperFunctionUtilsTest, SequenceSerialization) {
  std::vector<int32_t> V({1, 2, -47, 139});
  blobSerializationRoundTrip<SPSSequence<int32_t>, std::vector<int32_t>>(V);
}

TEST(WrapperFunctionUtilsTest, StringViewCharSequenceSerialization) {
  const char *HW = "Hello, world!";
  blobSerializationRoundTrip<SPSString, string_view>(string_view(HW));
}

TEST(WrapperFunctionUtilsTest, StdPairSerialization) {
  std::pair<int32_t, std::string> P(42, "foo");
  blobSerializationRoundTrip<SPSTuple<int32_t, SPSString>,
                             std::pair<int32_t, std::string>>(P);
}

TEST(WrapperFunctionUtilsTest, ArgListSerialization) {
  using BAL = SPSArgList<bool, int32_t, SPSString>;

  bool Arg1 = true;
  int32_t Arg2 = 42;
  std::string Arg3 = "foo";

  size_t Size = BAL::size(Arg1, Arg2, Arg3);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BAL::serialize(OB, Arg1, Arg2, Arg3));

  SPSInputBuffer IB(Buffer.get(), Size);

  bool ArgOut1;
  int32_t ArgOut2;
  std::string ArgOut3;

  EXPECT_TRUE(BAL::deserialize(IB, ArgOut1, ArgOut2, ArgOut3));

  EXPECT_EQ(Arg1, ArgOut1);
  EXPECT_EQ(Arg2, ArgOut2);
  EXPECT_EQ(Arg3, ArgOut3);
}

static __orc_rt_CWrapperFunctionResult addWrapper(const char *ArgData,
                                                  size_t ArgSize) {
  return WrapperFunction<int32_t(int32_t, int32_t)>::handle(
             ArgData, ArgSize,
             [](int32_t X, int32_t Y) -> int32_t { return X + Y; })
      .release();
}

extern "C" __orc_rt_Opaque __orc_rt_jit_dispatch_ctx{};

extern "C" __orc_rt_CWrapperFunctionResult
__orc_rt_jit_dispatch(__orc_rt_Opaque *Ctx, const void *FnTag,
                      const char *ArgData, size_t ArgSize) {
  using WrapperFunctionType =
      __orc_rt_CWrapperFunctionResult (*)(const char *, size_t);

  return reinterpret_cast<WrapperFunctionType>(const_cast<void *>(FnTag))(
      ArgData, ArgSize);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandle) {
  int32_t Result;
  EXPECT_FALSE(!!WrapperFunction<int32_t(int32_t, int32_t)>::call(
      (void *)&addWrapper, Result, 1, 2));
  EXPECT_EQ(Result, (int32_t)3);
}
