//===-- flang/unittests/Runtime/ListInputTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "../../runtime/io-error.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

// Pads characters with whitespace when needed
void SetCharacter(char *to, std::size_t n, const char *from) {
  auto len{std::strlen(from)};
  std::memcpy(to, from, std::min(len, n));
  if (len < n) {
    std::memset(to + len, ' ', n - len);
  }
}

struct InputTest : CrashHandlerFixture {};

TEST(InputTest, TestListInputAlphabet) {
  constexpr int numInputBuffers{2};
  constexpr int maxInputBufferLength{32};
  char inputBuffers[numInputBuffers][maxInputBufferLength];
  const char expectedOutput[]{
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "};
  int j{0};

  // Use _two_ input buffers and _three_ output buffers. Note the `3*` in the
  // _inputBuffers_.
  SetCharacter(inputBuffers[j++], maxInputBufferLength,
      "3*'abcdefghijklmnopqrstuvwxyzABC");
  SetCharacter(
      inputBuffers[j++], maxInputBufferLength, "DEFGHIJKLMNOPQRSTUVWXYZ'");

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{numInputBuffers};
  whole.Establish(TypeCode{CFI_type_char}, maxInputBufferLength, &inputBuffers,
      1, extent, CFI_attribute_pointer);
  whole.Check();
  auto *cookie{IONAME(BeginInternalArrayListInput)(whole)};

  constexpr int numOutputBuffers{3};
  constexpr int outputBufferLength{54};
  char outputBuffers[numOutputBuffers][outputBufferLength]{};
  for (j = 0; j < numOutputBuffers; ++j) {
    IONAME(InputAscii)(cookie, outputBuffers[j], outputBufferLength - 1);
  }

  const auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "list-directed input failed, status "
                       << static_cast<int>(status) << '\n';

  // Verify results that the _two_ ascii inputs result in _three_ alphabets
  for (j = 0; j < numOutputBuffers; ++j) {
    ASSERT_EQ(std::strcmp(outputBuffers[j], expectedOutput), 0)
        << "wanted outputBuffers[" << j << "]=" << expectedOutput << ", got '"
        << outputBuffers[j] << "'\n";
  }
}

TEST(InputTest, TestListInputIntegerList) {
  constexpr int numBuffers{2};
  constexpr int maxBufferLength{32};
  char buffer[numBuffers][maxBufferLength];
  int j{0};
  SetCharacter(buffer[j++], maxBufferLength, "1 2 2*3  ,");
  SetCharacter(buffer[j++], maxBufferLength, ",6,,8,2*");

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{numBuffers};
  whole.Establish(TypeCode{CFI_type_char}, maxBufferLength, &buffer, 1, extent,
      CFI_attribute_pointer);
  whole.Check();
  auto *cookie{IONAME(BeginInternalArrayListInput)(whole)};

  constexpr int listInputLength{10};

  // Negative numbers will be overwritten by _expectedOutput_, and positive
  // numbers will not be as their indices are "Null values" of the Fortran 2018
  // standard 13.10.3.2 in the format strings _buffer_
  std::int64_t actualOutput[listInputLength]{
      -1, -2, -3, -4, 5, -6, 7, -8, 9, 10};
  const std::int64_t expectedOutput[listInputLength]{
      1, 2, 3, 3, 5, 6, 7, 8, 9, 10};
  for (j = 0; j < listInputLength; ++j) {
    IONAME(InputInteger)(cookie, actualOutput[j]);
  }

  const auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "list-directed input failed, status "
                       << static_cast<int>(status) << '\n';

  // Verify the calls to _InputInteger_ resulted in _expectedOutput_
  for (j = 0; j < listInputLength; ++j) {
    ASSERT_EQ(actualOutput[j], expectedOutput[j])
        << "wanted actualOutput[" << j << "]==" << expectedOutput[j] << ", got "
        << actualOutput[j] << '\n';
  }
}

TEST(InputTest, TestListInputInvalidFormatWithSingleSuccess) {
  std::string formatBuffer{"1, g"};
  constexpr int numBuffers{1};

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{numBuffers};
  whole.Establish(TypeCode{CFI_type_char}, formatBuffer.size(),
      formatBuffer.data(), 1, extent, CFI_attribute_pointer);
  whole.Check();

  auto *cookie{IONAME(BeginInternalArrayListInput)(whole)};
  std::int64_t dummy;

  // Perform _InputInteger_ once successfully
  IONAME(InputInteger)(cookie, dummy);

  // Perform failing InputInteger
  ASSERT_DEATH(IONAME(InputInteger)(cookie, dummy),
      "Bad character 'g' in INTEGER input field");
}

// Same test as _TestListInputInvalidFormatWithSingleSuccess_, however no
// successful call to _InputInteger_ is performed first.
TEST(InputTest, TestListInputInvalidFormat) {
  std::string formatBuffer{"g"};
  constexpr int numBuffers{1};

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{numBuffers};
  whole.Establish(TypeCode{CFI_type_char}, formatBuffer.size(),
      formatBuffer.data(), 1, extent, CFI_attribute_pointer);
  whole.Check();

  auto *cookie{IONAME(BeginInternalArrayListInput)(whole)};
  std::int64_t dummy;

  // Perform failing InputInteger
  ASSERT_DEATH(IONAME(InputInteger)(cookie, dummy),
      "Bad character 'g' in INTEGER input field");
}

using ParamTy = std::tuple<std::string, std::vector<int>>;

struct SimpleListInputTest : testing::TestWithParam<ParamTy> {};

TEST_P(SimpleListInputTest, TestListInput) {
  auto [formatBuffer, expectedOutput] = GetParam();
  constexpr int numBuffers{1};

  StaticDescriptor<1> staticDescriptor;
  Descriptor &whole{staticDescriptor.descriptor()};
  SubscriptValue extent[]{numBuffers};
  whole.Establish(TypeCode{CFI_type_char}, formatBuffer.size(),
      formatBuffer.data(), 1, extent, CFI_attribute_pointer);
  whole.Check();
  auto *cookie{IONAME(BeginInternalArrayListInput)(whole)};

  const auto listInputLength{expectedOutput.size()};
  std::vector<std::int64_t> actualOutput(listInputLength);
  for (std::size_t j = 0; j < listInputLength; ++j) {
    IONAME(InputInteger)(cookie, actualOutput[j]);
  }

  const auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "list-directed input failed, status "
                       << static_cast<int>(status) << '\n';

  // Verify the calls to _InputInteger_ resulted in _expectedOutput_
  for (std::size_t j = 0; j < listInputLength; ++j) {
    ASSERT_EQ(actualOutput[j], expectedOutput[j])
        << "wanted actualOutput[" << j << "]==" << expectedOutput[j] << ", got "
        << actualOutput[j] << '\n';
  }
}

INSTANTIATE_TEST_SUITE_P(SimpleListInputTestInstantiation, SimpleListInputTest,
    testing::Values(std::make_tuple("", std::vector<int>{}),
        std::make_tuple("0", std::vector<int>{}),
        std::make_tuple("1", std::vector<int>{1}),
        std::make_tuple("1, 2", std::vector<int>{1, 2}),
        std::make_tuple("3*2", std::vector<int>{2, 2, 2})));
