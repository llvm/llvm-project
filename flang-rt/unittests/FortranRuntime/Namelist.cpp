//===-- flang/unittests/Runtime/Namelist.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/../../runtime/namelist.h"
#include "CrashHandlerFixture.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include "tools.h"
#include <algorithm>
#include <cinttypes>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

struct NamelistTests : CrashHandlerFixture {};

static void ClearDescriptorStorage(const Descriptor &descriptor) {
  std::memset(descriptor.raw().base_addr, 0,
              descriptor.Elements() * descriptor.ElementBytes());
}

TEST(NamelistTests, BasicSanity) {
  static constexpr int numLines{12};
  static constexpr int lineLength{32};
  static char buffer[numLines][lineLength];
  StaticDescriptor<1, true> statDescs[1];
  Descriptor &internalDesc{statDescs[0].descriptor()};
  SubscriptValue extent[]{numLines};
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/lineLength,
                         &buffer, 1, extent, CFI_attribute_pointer);
  // Set up data arrays
  std::vector<int> ints;
  for (int j{0}; j < 20; ++j) {
    ints.push_back(j % 2 == 0 ? (1 << j) : -(1 << j));
  }
  std::vector<double> reals{0.0,
                            -0.0,
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::lowest(),
                            std::numeric_limits<double>::epsilon()};
  std::vector<std::uint8_t> logicals;
  logicals.push_back(false);
  logicals.push_back(true);
  logicals.push_back(false);
  std::vector<std::complex<float>> complexes;
  complexes.push_back(std::complex<float>{123.0, -0.5});
  std::vector<std::string> characters;
  characters.emplace_back("aBcDeFgHiJkLmNoPqRsTuVwXyZ");
  characters.emplace_back("0123456789'\"..............");
  // Copy the data into new descriptors
  OwningPtr<Descriptor> intDesc{
      MakeArray<TypeCategory::Integer, static_cast<int>(sizeof(int))>(
          std::vector<int>{5, 4}, std::move(ints))};
  OwningPtr<Descriptor> realDesc{
      MakeArray<TypeCategory::Real, static_cast<int>(sizeof(double))>(
          std::vector<int>{4, 2}, std::move(reals))};
  OwningPtr<Descriptor> logicalDesc{
      MakeArray<TypeCategory::Logical, static_cast<int>(sizeof(std::uint8_t))>(
          std::vector<int>{3}, std::move(logicals))};
  OwningPtr<Descriptor> complexDesc{
      MakeArray<TypeCategory::Complex, static_cast<int>(sizeof(float))>(
          std::vector<int>{}, std::move(complexes))};
  OwningPtr<Descriptor> characterDesc{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{2}, std::move(characters), characters[0].size())};
  // Create a NAMELIST group
  static constexpr int items{5};
  const NamelistGroup::Item itemArray[items]{{"ints", *intDesc},
                                             {"reals", *realDesc},
                                             {"logicals", *logicalDesc},
                                             {"complexes", *complexDesc},
                                             {"characters", *characterDesc}};
  const NamelistGroup group{"group1", items, itemArray};
  // Do an internal NAMELIST write and check results
  auto outCookie1{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                       __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDelim)(outCookie1, "APOSTROPHE", 10));
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie1, group));
  auto outStatus1{IONAME(EndIoStatement)(outCookie1)};
  ASSERT_EQ(outStatus1, 0) << "Failed namelist output sanity, status "
                           << static_cast<int>(outStatus1);

  static const std::string expect{"&GROUP1 INTS= 1 -2 4 -8 16 -32  "
                                  " 64 -128 256 -512 1024 -2048    "
                                  " 4096 -8192 16384 -32768 65536  "
                                  " -131072 262144 -524288,REALS=  "
                                  " 0. -0. Inf -Inf NaN            "
                                  " 1.7976931348623157E+308        "
                                  " -1.7976931348623157E+308       "
                                  " 2.220446049250313E-16,LOGICALS="
                                  "F T F,COMPLEXES= (123.,-.5),    "
                                  " CHARACTERS= 'aBcDeFgHiJkLmNoPqR"
                                  "sTuVwXyZ' '0123456789''\"........"
                                  "......'/                        "};
  std::string got{buffer[0], sizeof buffer};
  EXPECT_EQ(got, expect);

  // Clear the arrays, read them back, write out again, and compare
  ClearDescriptorStorage(*intDesc);
  ClearDescriptorStorage(*realDesc);
  ClearDescriptorStorage(*logicalDesc);
  ClearDescriptorStorage(*complexDesc);
  ClearDescriptorStorage(*characterDesc);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  auto inStatus{IONAME(EndIoStatement)(inCookie)};
  ASSERT_EQ(inStatus, 0) << "Failed namelist input sanity, status "
                         << static_cast<int>(inStatus);
  auto outCookie2{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                       __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDelim)(outCookie2, "APOSTROPHE", 10));
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie2, group));
  auto outStatus2{IONAME(EndIoStatement)(outCookie2)};
  ASSERT_EQ(outStatus2, 0) << "Failed namelist output sanity rewrite, status "
                           << static_cast<int>(outStatus2);
  std::string got2{buffer[0], sizeof buffer};
  EXPECT_EQ(got2, expect);
}

TEST(NamelistTests, Subscripts) {
  // INTEGER :: A(-1:0, -1:1)
  OwningPtr<Descriptor> aDesc{
      MakeArray<TypeCategory::Integer, static_cast<int>(sizeof(int))>(
          std::vector<int>{2, 3}, std::vector<int>(6, 0))};
  aDesc->GetDimension(0).SetBounds(-1, 0);
  aDesc->GetDimension(1).SetBounds(-1, 1);
  const NamelistGroup::Item items[]{{"a", *aDesc}};
  const NamelistGroup group{"justa", 1, items};
  static char t1[]{"&justa A(0,+1:-1:-2)=1 2/"};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  internalDesc.Establish(TypeCode{CFI_type_char},
                         /*elementBytes=*/std::strlen(t1), t1, 0, nullptr,
                         CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  auto inStatus{IONAME(EndIoStatement)(inCookie)};
  ASSERT_EQ(inStatus, 0) << "Failed namelist input subscripts, status "
                         << static_cast<int>(inStatus);
  char out[40];
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/sizeof out,
                         out, 0, nullptr, CFI_attribute_pointer);
  auto outCookie{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                      __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie, group));
  auto outStatus{IONAME(EndIoStatement)(outCookie)};
  ASSERT_EQ(outStatus, 0)
      << "Failed namelist output subscripts rewrite, status "
      << static_cast<int>(outStatus);
  std::string got{out, sizeof out};
  static const std::string expect{"&JUSTA A= 0 2 0 0 0 1/                  "};
  EXPECT_EQ(got, expect);
}

TEST(NamelistTests, ShortArrayInput) {
  OwningPtr<Descriptor> aDesc{
      MakeArray<TypeCategory::Integer, static_cast<int>(sizeof(int))>(
          std::vector<int>{2}, std::vector<int>(2, -1))};
  OwningPtr<Descriptor> bDesc{
      MakeArray<TypeCategory::Integer, static_cast<int>(sizeof(int))>(
          std::vector<int>{2}, std::vector<int>(2, -2))};
  const NamelistGroup::Item items[]{{"a", *aDesc}, {"b", *bDesc}};
  const NamelistGroup group{"nl", 2, items};
  // Two 12-character lines of internal input
  static char t1[]{"&nl a = 1 b "
                   " = 2 /      "};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  SubscriptValue shape{2};
  internalDesc.Establish(1, 12, t1, 1, &shape, CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  auto inStatus{IONAME(EndIoStatement)(inCookie)};
  ASSERT_EQ(inStatus, 0) << "Failed namelist input subscripts, status "
                         << static_cast<int>(inStatus);
  EXPECT_EQ(*aDesc->ZeroBasedIndexedElement<int>(0), 1);
  EXPECT_EQ(*aDesc->ZeroBasedIndexedElement<int>(1), -1);
  EXPECT_EQ(*bDesc->ZeroBasedIndexedElement<int>(0), 2);
  EXPECT_EQ(*bDesc->ZeroBasedIndexedElement<int>(1), -2);
}

TEST(NamelistTests, ScalarSubstring) {
  OwningPtr<Descriptor> scDesc{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{}, std::vector<std::string>{"abcdefgh"}, 8)};
  const NamelistGroup::Item items[]{{"a", *scDesc}};
  const NamelistGroup group{"justa", 1, items};
  static char t1[]{"&justa A(2:5)='BCDE'/"};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  internalDesc.Establish(TypeCode{CFI_type_char},
                         /*elementBytes=*/std::strlen(t1), t1, 0, nullptr,
                         CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(inCookie), IostatOk)
      << "namelist scalar substring input";
  char out[32];
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/sizeof out,
                         out, 0, nullptr, CFI_attribute_pointer);
  auto outCookie{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                      __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDelim)(outCookie, "apostrophe", 10));
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(outCookie), IostatOk) << "namelist output";
  std::string got{out, sizeof out};
  static const std::string expect{"&JUSTA A= 'aBCDEfgh'/           "};
  EXPECT_EQ(got, expect);
}

TEST(NamelistTests, ArraySubstring) {
  OwningPtr<Descriptor> scDesc{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{2}, std::vector<std::string>{"abcdefgh", "ijklmnop"},
      8)};
  const NamelistGroup::Item items[]{{"a", *scDesc}};
  const NamelistGroup group{"justa", 1, items};
  static char t1[]{"&justa A(:)(2:+5)='BCDE' 'JKLM'/"};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  internalDesc.Establish(TypeCode{CFI_type_char},
                         /*elementBytes=*/std::strlen(t1), t1, 0, nullptr,
                         CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(inCookie), IostatOk)
      << "namelist scalar substring input";
  char out[40];
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/sizeof out,
                         out, 0, nullptr, CFI_attribute_pointer);
  auto outCookie{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                      __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDelim)(outCookie, "apostrophe", 10));
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(outCookie), IostatOk) << "namelist output";
  std::string got{out, sizeof out};
  static const std::string expect{"&JUSTA A= 'aBCDEfgh' 'iJKLMnop'/        "};
  EXPECT_EQ(got, expect);
}

TEST(NamelistTests, Skip) {
  OwningPtr<Descriptor> scDesc{
      MakeArray<TypeCategory::Integer, static_cast<int>(sizeof(int))>(
          std::vector<int>{}, std::vector<int>{-1})};
  const NamelistGroup::Item items[]{{"j", *scDesc}};
  const NamelistGroup group{"nml", 1, items};
  static char t1[]{"&skip a='str''ing'/&nml j=123/"};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  internalDesc.Establish(TypeCode{CFI_type_char},
                         /*elementBytes=*/std::strlen(t1), t1, 0, nullptr,
                         CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(inCookie), IostatOk)
      << "namelist input with skipping";
  char out[20];
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/sizeof out,
                         out, 0, nullptr, CFI_attribute_pointer);
  auto outCookie{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                      __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(outCookie), IostatOk) << "namelist output";
  std::string got{out, sizeof out};
  static const std::string expect{"&NML J= 123/        "};
  EXPECT_EQ(got, expect);
}

// Tests DECIMAL=COMMA mode
TEST(NamelistTests, Comma) {
  OwningPtr<Descriptor> scDesc{
      MakeArray<TypeCategory::Complex, static_cast<int>(sizeof(float))>(
          std::vector<int>{2}, std::vector<std::complex<float>>{{}, {}})};
  const NamelistGroup::Item items[]{{"z", *scDesc}};
  const NamelistGroup group{"nml", 1, items};
  static char t1[]{"&nml z=(-1,0;2,0);(-3,0;0,5)/"};
  StaticDescriptor<1, true> statDesc;
  Descriptor &internalDesc{statDesc.descriptor()};
  internalDesc.Establish(TypeCode{CFI_type_char},
                         /*elementBytes=*/std::strlen(t1), t1, 0, nullptr,
                         CFI_attribute_pointer);
  auto inCookie{IONAME(BeginInternalArrayListInput)(internalDesc, nullptr, 0,
                                                    __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDecimal)(inCookie, "COMMA", 5));
  ASSERT_TRUE(IONAME(InputNamelist)(inCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(inCookie), IostatOk)
      << "namelist input with skipping";
  char out[30];
  internalDesc.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/sizeof out,
                         out, 0, nullptr, CFI_attribute_pointer);
  auto outCookie{IONAME(BeginInternalArrayListOutput)(internalDesc, nullptr, 0,
                                                      __FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetDecimal)(outCookie, "COMMA", 5));
  ASSERT_TRUE(IONAME(OutputNamelist)(outCookie, group));
  ASSERT_EQ(IONAME(EndIoStatement)(outCookie), IostatOk) << "namelist output";
  std::string got{out, sizeof out};
  static const std::string expect{"&NML Z= (-1,;2,) (-3,;,5)/    "};
  EXPECT_EQ(got, expect);
}

// TODO: Internal NAMELIST error tests
