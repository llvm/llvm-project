//===-- flang/unittests/Runtime/NumericalFormatTest.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <tuple>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

static bool CompareFormattedStrings(
    const std::string &expect, const std::string &got) {
  std::string want{expect};
  want.resize(got.size(), ' ');
  return want == got;
}

static bool CompareFormattedStrings(
    const char *expect, const std::string &&got) {
  return CompareFormattedStrings(std::string(expect), std::move(got));
}

// Perform format and compare the result with expected value
static bool CompareFormatReal(
    const char *format, double x, const char *expect, std::string &got) {
  char buffer[800];
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  EXPECT_TRUE(IONAME(OutputReal64)(cookie, x));
  auto status{IONAME(EndIoStatement)(cookie)};
  EXPECT_EQ(status, 0);
  got = std::string{buffer, sizeof buffer};
  auto lastNonBlank{got.find_last_not_of(" ")};
  if (lastNonBlank != std::string::npos) {
    got.resize(lastNonBlank + 1);
  }
  return CompareFormattedStrings(expect, got);
}

// Convert raw uint64 into double, perform format, and compare with expected
static bool CompareFormatReal(const char *format, std::uint64_t xInt,
    const char *expect, std::string &got) {
  double x;
  static_assert(sizeof(double) == sizeof(std::uint64_t),
      "Size of double != size of uint64_t!");
  std::memcpy(&x, &xInt, sizeof xInt);
  return CompareFormatReal(format, x, expect, got);
}

static bool CompareFormatInteger(
    const char *format, std::int64_t x, const char *expect, std::string &got) {
  char buffer[800];
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  EXPECT_TRUE(IONAME(OutputInteger64)(cookie, x));
  auto status{IONAME(EndIoStatement)(cookie)};
  EXPECT_EQ(status, 0);
  got = std::string{buffer, sizeof buffer};
  auto lastNonBlank{got.find_last_not_of(" ")};
  if (lastNonBlank != std::string::npos) {
    got.resize(lastNonBlank + 1);
  }
  return CompareFormattedStrings(expect, got);
}

struct IOApiTests : CrashHandlerFixture {};

TEST(IOApiTests, HelloWorldOutputTest) {
  static constexpr int bufferSize{32};
  char buffer[bufferSize];

  // Create format for all types and values to be written
  const char *format{"(6HHELLO,,A6,2X,I3,1X,'0x',Z8,1X,L1)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, bufferSize, format, std::strlen(format))};

  // Write string, integer, and logical values to buffer
  IONAME(OutputAscii)(cookie, "WORLD", 5);
  IONAME(OutputInteger64)(cookie, 678);
  IONAME(OutputInteger32)(cookie, 0xfeedface);
  IONAME(OutputLogical)(cookie, true);

  // Ensure IO succeeded
  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "hello: '" << format << "' failed, status "
                       << static_cast<int>(status);

  // Ensure final buffer matches expected string output
  static const std::string expect{"HELLO, WORLD  678 0xFEEDFACE T"};
  ASSERT_TRUE(
      CompareFormattedStrings(expect, std::string{buffer, sizeof buffer}))
      << "Expected '" << expect << "', got " << buffer;
}

TEST(IOApiTests, MultilineOutputTest) {
  // Allocate buffer for multiline output
  static constexpr int numLines{5};
  static constexpr int lineLength{32};
  char buffer[numLines][lineLength];

  // Create descriptor for entire buffer
  static constexpr int staticDescriptorMaxRank{1};
  StaticDescriptor<staticDescriptorMaxRank> wholeStaticDescriptor;
  Descriptor &whole{wholeStaticDescriptor.descriptor()};
  static const SubscriptValue extent[]{numLines};
  whole.Establish(TypeCode{CFI_type_char}, /*elementBytes=*/lineLength, &buffer,
      staticDescriptorMaxRank, extent, CFI_attribute_pointer);
  whole.Dump(stderr);
  whole.Check();

  // Create descriptor for buffer section
  StaticDescriptor<staticDescriptorMaxRank> sectionStaticDescriptor;
  Descriptor &section{sectionStaticDescriptor.descriptor()};
  static const SubscriptValue lowers[]{0}, uppers[]{4}, strides[]{1};
  section.Establish(whole.type(), /*elementBytes=*/whole.ElementBytes(),
      nullptr, /*maxRank=*/staticDescriptorMaxRank, extent,
      CFI_attribute_pointer);

  // Ensure C descriptor address `section.raw()` is updated without error
  const auto error{
      CFI_section(&section.raw(), &whole.raw(), lowers, uppers, strides)};
  ASSERT_EQ(error, 0) << "multiline: CFI_section failed: " << error;
  section.Dump(stderr);
  section.Check();

  // Create format string and initialize IO operation
  const char *format{
      "('?abcde,',T1,'>',T9,A,TL12,A,TR25,'<'//G0,17X,'abcd',1(2I4))"};
  auto cookie{IONAME(BeginInternalArrayFormattedOutput)(
      section, format, std::strlen(format))};

  // Fill last line with periods
  std::memset(buffer[numLines - 1], '.', lineLength);

  // Write data to buffer
  IONAME(OutputAscii)(cookie, "WORLD", 5);
  IONAME(OutputAscii)(cookie, "HELLO", 5);
  IONAME(OutputInteger64)(cookie, 789);
  for (int j{666}; j <= 999; j += 111) {
    IONAME(OutputInteger64)(cookie, j);
  }

  // Ensure no errors occured in write operations above
  const auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "multiline: '" << format << "' failed, status "
                       << static_cast<int>(status);

  static const std::string expect{">HELLO, WORLD                  <"
                                  "                                "
                                  "789                 abcd 666 777"
                                  " 888 999                        "
                                  "................................"};
  // Ensure formatted string matches expected output
  EXPECT_TRUE(
      CompareFormattedStrings(expect, std::string{buffer[0], sizeof buffer}))
      << "Expected '" << expect << "' but got '"
      << std::string{buffer[0], sizeof buffer} << "'";
}

TEST(IOApiTests, ListInputTest) {
  static const char input[]{",1*,(5.,6.),(7.0,8.0)"};
  auto cookie{IONAME(BeginInternalListInput)(input, sizeof input - 1)};

  // Create real values for IO tests
  static constexpr int numRealValues{8};
  float z[numRealValues];
  for (int j{0}; j < numRealValues; ++j) {
    z[j] = -(j + 1);
  }

  // Ensure reading complex values to floats does not result in an error
  for (int j{0}; j < numRealValues; j += 2) {
    ASSERT_TRUE(IONAME(InputComplex32)(cookie, &z[j]))
        << "InputComplex32 failed with value " << z[j];
  }

  // Ensure no IO errors occured during IO operations above
  auto status{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(status, 0) << "Failed complex list-directed input, status "
                       << static_cast<int>(status);

  // Ensure writing complex values from floats does not result in an error
  static constexpr int bufferSize{39};
  char output[bufferSize];
  output[bufferSize - 1] = '\0';
  cookie = IONAME(BeginInternalListOutput)(output, bufferSize - 1);
  for (int j{0}; j < numRealValues; j += 2) {
    ASSERT_TRUE(IONAME(OutputComplex32)(cookie, z[j], z[j + 1]))
        << "OutputComplex32 failed when outputting value " << z[j] << ", "
        << z[j + 1];
  }

  // Ensure no IO errors occured during IO operations above
  status = IONAME(EndIoStatement)(cookie);
  ASSERT_EQ(status, 0) << "Failed complex list-directed output, status "
                       << static_cast<int>(status);

  // Verify output buffer against expected value
  static const char expect[bufferSize]{
      " (-1.,-2.) (-3.,-4.) (5.,6.) (7.,8.)  "};
  ASSERT_EQ(std::strncmp(output, expect, bufferSize), 0)
      << "Failed complex list-directed output, expected '" << expect
      << "', but got '" << output << "'";
}

TEST(IOApiTests, DescriptorOutputTest) {
  static constexpr int bufferSize{10};
  char buffer[bufferSize];
  const char *format{"(2A4)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, bufferSize, format, std::strlen(format))};

  // Create descriptor for output
  static constexpr int staticDescriptorMaxRank{1};
  StaticDescriptor<staticDescriptorMaxRank> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  static constexpr int subscriptExtent{2};
  static const SubscriptValue extent[]{subscriptExtent};

  // Manually write to descriptor buffer
  static constexpr int dataLength{4};
  char data[subscriptExtent][dataLength];
  std::memcpy(data[0], "ABCD", dataLength);
  std::memcpy(data[1], "EFGH", dataLength);
  desc.Establish(TypeCode{CFI_type_char}, dataLength, &data,
      staticDescriptorMaxRank, extent);
  desc.Dump(stderr);
  desc.Check();
  IONAME(OutputDescriptor)(cookie, desc);

  // Ensure no errors were encountered in initializing the cookie and descriptor
  auto formatStatus{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(formatStatus, 0)
      << "descrOutputTest: '" << format << "' failed, status "
      << static_cast<int>(formatStatus);

  // Ensure buffer matches expected output
  EXPECT_TRUE(
      CompareFormattedStrings("ABCDEFGH  ", std::string{buffer, sizeof buffer}))
      << "descrOutputTest: formatted: got '"
      << std::string{buffer, sizeof buffer} << "'";

  // Begin list-directed output on cookie by descriptor
  cookie = IONAME(BeginInternalListOutput)(buffer, sizeof buffer);
  IONAME(OutputDescriptor)(cookie, desc);

  // Ensure list-directed output does not result in an IO error
  auto listDirectedStatus{IONAME(EndIoStatement)(cookie)};
  ASSERT_EQ(listDirectedStatus, 0)
      << "descrOutputTest: list-directed failed, status "
      << static_cast<int>(listDirectedStatus);

  // Ensure buffer matches expected output
  EXPECT_TRUE(
      CompareFormattedStrings(" ABCDEFGH ", std::string{buffer, sizeof buffer}))
      << "descrOutputTest: list-directed: got '"
      << std::string{buffer, sizeof buffer} << "'";
}

//------------------------------------------------------------------------------
/// Tests for output formatting real values
//------------------------------------------------------------------------------

TEST(IOApiTests, FormatZeroes) {
  static constexpr std::pair<const char *, const char *> zeroes[]{
      {"(E32.17,';')", "         0.00000000000000000E+00;"},
      {"(F32.17,';')", "             0.00000000000000000;"},
      {"(G32.17,';')", "          0.0000000000000000    ;"},
      {"(DC,E32.17,';')", "         0,00000000000000000E+00;"},
      {"(DC,F32.17,';')", "             0,00000000000000000;"},
      {"(DC,G32.17,';')", "          0,0000000000000000    ;"},
      {"(D32.17,';')", "         0.00000000000000000D+00;"},
      {"(E32.17E1,';')", "          0.00000000000000000E+0;"},
      {"(G32.17E1,';')", "           0.0000000000000000   ;"},
      {"(E32.17E0,';')", "          0.00000000000000000E+0;"},
      {"(G32.17E0,';')", "          0.0000000000000000    ;"},
      {"(1P,E32.17,';')", "         0.00000000000000000E+00;"},
      {"(1PE32.17,';')", "         0.00000000000000000E+00;"}, // no comma
      {"(1P,F32.17,';')", "             0.00000000000000000;"},
      {"(1P,G32.17,';')", "          0.0000000000000000    ;"},
      {"(2P,E32.17,';')", "         00.0000000000000000E+00;"},
      {"(-1P,E32.17,';')", "         0.00000000000000000E+00;"},
      {"(G0,';')", "0.;"},
  };

  for (auto const &[format, expect] : zeroes) {
    std::string got;
    ASSERT_TRUE(CompareFormatReal(format, 0.0, expect, got))
        << "Failed to format " << format << ", expected '" << expect
        << "', got '" << got << "'";
  }
}

TEST(IOApiTests, FormatOnes) {
  static constexpr std::pair<const char *, const char *> ones[]{
      {"(E32.17,';')", "         0.10000000000000000E+01;"},
      {"(F32.17,';')", "             1.00000000000000000;"},
      {"(G32.17,';')", "          1.0000000000000000    ;"},
      {"(E32.17E1,';')", "          0.10000000000000000E+1;"},
      {"(G32.17E1,';')", "           1.0000000000000000   ;"},
      {"(E32.17E0,';')", "          0.10000000000000000E+1;"},
      {"(G32.17E0,';')", "          1.0000000000000000    ;"},
      {"(E32.17E4,';')", "       0.10000000000000000E+0001;"},
      {"(G32.17E4,';')", "        1.0000000000000000      ;"},
      {"(1P,E32.17,';')", "         1.00000000000000000E+00;"},
      {"(1PE32.17,';')", "         1.00000000000000000E+00;"}, // no comma
      {"(1P,F32.17,';')", "            10.00000000000000000;"},
      {"(1P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(ES32.17,';')", "         1.00000000000000000E+00;"},
      {"(2P,E32.17,';')", "         10.0000000000000000E-01;"},
      {"(2P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(-1P,E32.17,';')", "         0.01000000000000000E+02;"},
      {"(-1P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(G0,';')", "1.;"},
  };

  for (auto const &[format, expect] : ones) {
    std::string got;
    ASSERT_TRUE(CompareFormatReal(format, 1.0, expect, got))
        << "Failed to format " << format << ", expected '" << expect
        << "', got '" << got << "'";
  }
}

TEST(IOApiTests, FormatNegativeOnes) {
  static constexpr std::tuple<const char *, const char *> negOnes[]{
      {"(E32.17,';')", "        -0.10000000000000000E+01;"},
      {"(F32.17,';')", "            -1.00000000000000000;"},
      {"(G32.17,';')", "         -1.0000000000000000    ;"},
      {"(G0,';')", "-1.;"},
  };
  for (auto const &[format, expect] : negOnes) {
    std::string got;
    ASSERT_TRUE(CompareFormatReal(format, -1.0, expect, got))
        << "Failed to format " << format << ", expected '" << expect
        << "', got '" << got << "'";
  }
}

// Each test case contains a raw uint64, a format string for a real value, and
// the expected resulting string from formatting the raw uint64. The double
// representation of the uint64 is commented above each test case.
TEST(IOApiTests, FormatDoubleValues) {

  using TestCaseTy = std::tuple<std::uint64_t,
      std::vector<std::tuple<const char *, const char *>>>;
  static const std::vector<TestCaseTy> testCases{
      {// -0
          0x8000000000000000,
          {
              {"(E9.1,';')", " -0.0E+00;"},
              {"(F4.0,';')", " -0.;"},
              {"(F0.1,';')", "-.0;"},
              {"(G8.0,';')", "-0.0E+00;"},
              {"(G8.1,';')", " -0.    ;"},
              {"(G0,';')", "-0.;"},
              {"(E9.1,';')", " -0.0E+00;"},
          }},
      {// +Inf
          0x7ff0000000000000,
          {
              {"(E9.1,';')", "      Inf;"},
              {"(F9.1,';')", "      Inf;"},
              {"(G9.1,';')", "      Inf;"},
              {"(SP,E9.1,';')", "     +Inf;"},
              {"(SP,F9.1,';')", "     +Inf;"},
              {"(SP,G9.1,';')", "     +Inf;"},
              {"(G0,';')", "Inf;"},
          }},
      {// -Inf
          0xfff0000000000000,
          {
              {"(E9.1,';')", "     -Inf;"},
              {"(F9.1,';')", "     -Inf;"},
              {"(G9.1,';')", "     -Inf;"},
              {"(G0,';')", "-Inf;"},
          }},
      {// NaN
          0x7ff0000000000001,
          {
              {"(E9.1,';')", "      NaN;"},
              {"(F9.1,';')", "      NaN;"},
              {"(G9.1,';')", "      NaN;"},
              {"(G0,';')", "NaN;"},
          }},
      {// NaN (sign irrelevant)
          0xfff0000000000001,
          {
              {"(E9.1,';')", "      NaN;"},
              {"(F9.1,';')", "      NaN;"},
              {"(G9.1,';')", "      NaN;"},
              {"(SP,E9.1,';')", "      NaN;"},
              {"(SP,F9.1,';')", "      NaN;"},
              {"(SP,G9.1,';')", "      NaN;"},
              {"(G0,';')", "NaN;"},
          }},
      {// 0.1 rounded
          0x3fb999999999999a,
          {
              {"(E62.55,';')",
                  " 0.1000000000000000055511151231257827021181583404541015625E+"
                  "00;"},
              {"(E0.0,';')", ".1E+00;"},
              {"(E0.55,';')",
                  ".1000000000000000055511151231257827021181583404541015625E+"
                  "00;"},
              {"(E0,';')", ".1E+00;"},
              {"(F58.55,';')",
                  " 0."
                  "1000000000000000055511151231257827021181583404541015625;"},
              {"(F0.0,';')", "0.;"},
              {"(F0.55,';')",
                  ".1000000000000000055511151231257827021181583404541015625;"},
              {"(F0,';')", ".1;"},
              {"(G62.55,';')",
                  " 0.1000000000000000055511151231257827021181583404541015625  "
                  "  ;"},
              {"(G0.0,';')", "0.;"},
              {"(G0.55,';')",
                  ".1000000000000000055511151231257827021181583404541015625;"},
              {"(G0,';')", ".1;"},
          }},
      {// 1.5
          0x3ff8000000000000,
          {
              {"(E9.2,';')", " 0.15E+01;"},
              {"(F4.1,';')", " 1.5;"},
              {"(G7.1,';')", " 2.    ;"},
              {"(RN,E8.1,';')", " 0.2E+01;"},
              {"(RN,F3.0,';')", " 2.;"},
              {"(RN,G7.0,';')", " 0.E+01;"},
              {"(RN,G7.1,';')", " 2.    ;"},
              {"(RD,E8.1,';')", " 0.1E+01;"},
              {"(RD,F3.0,';')", " 1.;"},
              {"(RD,G7.0,';')", " 0.E+01;"},
              {"(RD,G7.1,';')", " 1.    ;"},
              {"(RU,E8.1,';')", " 0.2E+01;"},
              {"(RU,G7.0,';')", " 0.E+01;"},
              {"(RU,G7.1,';')", " 2.    ;"},
              {"(RZ,E8.1,';')", " 0.1E+01;"},
              {"(RZ,F3.0,';')", " 1.;"},
              {"(RZ,G7.0,';')", " 0.E+01;"},
              {"(RZ,G7.1,';')", " 1.    ;"},
              {"(RC,E8.1,';')", " 0.2E+01;"},
              {"(RC,F3.0,';')", " 2.;"},
              {"(RC,G7.0,';')", " 0.E+01;"},
              {"(RC,G7.1,';')", " 2.    ;"},
          }},
      {// -1.5
          0xbff8000000000000,
          {
              {"(E9.2,';')", "-0.15E+01;"},
              {"(RN,E8.1,';')", "-0.2E+01;"},
              {"(RD,E8.1,';')", "-0.2E+01;"},
              {"(RU,E8.1,';')", "-0.1E+01;"},
              {"(RZ,E8.1,';')", "-0.1E+01;"},
              {"(RC,E8.1,';')", "-0.2E+01;"},
          }},
      {// 2.5
          0x4004000000000000,
          {
              {"(E9.2,';')", " 0.25E+01;"},
              {"(RN,E8.1,';')", " 0.2E+01;"},
              {"(RD,E8.1,';')", " 0.2E+01;"},
              {"(RU,E8.1,';')", " 0.3E+01;"},
              {"(RZ,E8.1,';')", " 0.2E+01;"},
              {"(RC,E8.1,';')", " 0.3E+01;"},
          }},
      {// -2.5
          0xc004000000000000,
          {
              {"(E9.2,';')", "-0.25E+01;"},
              {"(RN,E8.1,';')", "-0.2E+01;"},
              {"(RD,E8.1,';')", "-0.3E+01;"},
              {"(RU,E8.1,';')", "-0.2E+01;"},
              {"(RZ,E8.1,';')", "-0.2E+01;"},
              {"(RC,E8.1,';')", "-0.3E+01;"},
          }},
      {// least positive nonzero subnormal
          1,
          {
              {"(E32.17,';')", "         0.49406564584124654-323;"},
              {"(ES32.17,';')", "         4.94065645841246544-324;"},
              {"(EN32.17,';')", "         4.94065645841246544-324;"},
              {"(E759.752,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "75386825064197182655334472656250-323;"},
              {"(G0,';')", ".5E-323;"},
              {"(E757.750,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "753868250641971826553344726562-323;"},
              {"(RN,E757.750,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "753868250641971826553344726562-323;"},
              {"(RD,E757.750,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "753868250641971826553344726562-323;"},
              {"(RU,E757.750,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "753868250641971826553344726563-323;"},
              {"(RC,E757.750,';')",
                  " 0."
                  "494065645841246544176568792868221372365059802614324764425585"
                  "682500675507270208751865299836361635992379796564695445717730"
                  "926656710355939796398774796010781878126300713190311404527845"
                  "817167848982103688718636056998730723050006387409153564984387"
                  "312473397273169615140031715385398074126238565591171026658556"
                  "686768187039560310624931945271591492455329305456544401127480"
                  "129709999541931989409080416563324524757147869014726780159355"
                  "238611550134803526493472019379026810710749170333222684475333"
                  "572083243193609238289345836806010601150616980975307834227731"
                  "832924790498252473077637592724787465608477820373446969953364"
                  "701797267771758512566055119913150489110145103786273816725095"
                  "583738973359899366480994116420570263709027924276754456522908"
                  "753868250641971826553344726563-323;"},
          }},
      {// least positive nonzero normal
          0x10000000000000,
          {
              {"(E723.716,';')",
                  " 0."
                  "222507385850720138309023271733240406421921598046233183055332"
                  "741688720443481391819585428315901251102056406733973103581100"
                  "515243416155346010885601238537771882113077799353200233047961"
                  "014744258363607192156504694250373420837525080665061665815894"
                  "872049117996859163964850063590877011830487479978088775374994"
                  "945158045160505091539985658247081864511353793580499211598108"
                  "576605199243335211435239014879569960959128889160299264151106"
                  "346631339366347758651302937176204732563178148566435087212282"
                  "863764204484681140761391147706280168985324411002416144742161"
                  "856716615054015428508471675290190316132277889672970737312333"
                  "408698898317506783884692609277397797285865965494109136909540"
                  "61364675687023986783152906809846172109246253967285156250-"
                  "307;"},
              {"(G0,';')", ".22250738585072014E-307;"},
          }},
      {// greatest finite
          0x7fefffffffffffffuLL,
          {
              {"(E32.17,';')", "         0.17976931348623157+309;"},
              {"(E317.310,';')",
                  " 0."
                  "179769313486231570814527423731704356798070567525844996598917"
                  "476803157260780028538760589558632766878171540458953514382464"
                  "234321326889464182768467546703537516986049910576551282076245"
                  "490090389328944075868508455133942304583236903222948165808559"
                  "332123348274797826204144723168738177180919299881250404026184"
                  "1248583680+309;"},
              {"(ES317.310,';')",
                  " 1."
                  "797693134862315708145274237317043567980705675258449965989174"
                  "768031572607800285387605895586327668781715404589535143824642"
                  "343213268894641827684675467035375169860499105765512820762454"
                  "900903893289440758685084551339423045832369032229481658085593"
                  "321233482747978262041447231687381771809192998812504040261841"
                  "2485836800+308;"},
              {"(EN319.310,';')",
                  " 179."
                  "769313486231570814527423731704356798070567525844996598917476"
                  "803157260780028538760589558632766878171540458953514382464234"
                  "321326889464182768467546703537516986049910576551282076245490"
                  "090389328944075868508455133942304583236903222948165808559332"
                  "123348274797826204144723168738177180919299881250404026184124"
                  "8583680000+306;"},
              {"(G0,';')", ".17976931348623157E+309;"},
          }},
  };

  for (auto const &[value, cases] : testCases) {
    for (auto const &[format, expect] : cases) {
      std::string got;
      ASSERT_TRUE(CompareFormatReal(format, value, expect, got))
          << "Failed to format " << format << ", expected '" << expect
          << "', got '" << got << "'";
    }
  }

  using IndividualTestCaseTy = std::tuple<const char *, double, const char *>;
  static const std::vector<IndividualTestCaseTy> individualTestCases{
      {"(F5.3,';')", 25., "*****;"},
      {"(F5.3,';')", 2.5, "2.500;"},
      {"(F5.3,';')", 0.25, "0.250;"},
      {"(F5.3,';')", 0.025, "0.025;"},
      {"(F5.3,';')", 0.0025, "0.003;"},
      {"(F5.3,';')", 0.00025, "0.000;"},
      {"(F5.3,';')", 0.000025, "0.000;"},
      {"(F5.3,';')", -25., "*****;"},
      {"(F5.3,';')", -2.5, "*****;"},
      {"(F5.3,';')", -0.25, "-.250;"},
      {"(F5.3,';')", -0.025, "-.025;"},
      {"(F5.3,';')", -0.0025, "-.003;"},
      {"(F5.3,';')", -0.00025, "-.000;"},
      {"(F5.3,';')", -0.000025, "-.000;"},
      {"(F5.3,';')", 99.999, "*****;"},
      {"(F5.3,';')", 9.9999, "*****;"},
      {"(F5.3,';')", 0.99999, "1.000;"},
      {"(F5.3,';')", 0.099999, "0.100;"},
      {"(F5.3,';')", 0.0099999, "0.010;"},
      {"(F5.3,';')", 0.00099999, "0.001;"},
      {"(F5.3,';')", 0.0005, "0.001;"},
      {"(F5.3,';')", 0.00049999, "0.000;"},
      {"(F5.3,';')", 0.000099999, "0.000;"},
      {"(F5.3,';')", -99.999, "*****;"},
      {"(F5.3,';')", -9.9999, "*****;"},
      {"(F5.3,';')", -0.99999, "*****;"},
      {"(F5.3,';')", -0.099999, "-.100;"},
      {"(F5.3,';')", -0.0099999, "-.010;"},
      {"(F5.3,';')", -0.00099999, "-.001;"},
      {"(F5.3,';')", -0.0005, "-.001;"},
      {"(F5.3,';')", -0.00049999, "-.000;"},
      {"(F5.3,';')", -0.000099999, "-.000;"},
      {"(F0.1,';')", 0.0, ".0;"},
  };

  for (auto const &[format, value, expect] : individualTestCases) {
    std::string got;
    ASSERT_TRUE(CompareFormatReal(format, value, expect, got))
        << "Failed to format " << format << ", expected '" << expect
        << "', got '" << got << "'";
  }

  // Problematic EN formatting edge cases with rounding
  using IndividualENTestCaseTy = std::tuple<std::uint64_t, const char *>;
  static const std::vector<IndividualENTestCaseTy> individualENTestCases{
      {0x3E11183197785F8C, " 995.0E-12"}, // 0.9950312500000000115852E-09
      {0x3E11180E68455D30, " 995.0E-12"}, // 0.9949999999999999761502E-09
      {0x3E112BD8F4F6B0D7, " 999.5E-12"}, // 0.9994999999999999089118E-09
      {0x3E45794883CA8782, "  10.0E-09"}, // 0.9999499999999999642266E-08
      {0x3F506218230C7482, " 999.9E-06"}, // 0.9999499999999998840761E-03
      {0x3FB99652BD3C3612, " 100.0E-03"}, // 0.9999500000000000055067E+00
      {0x4023E66666666667, "  10.0E+00"}, // 0.9950000000000001065814E+01
  };

  for (auto const &[value, expect] : individualENTestCases) {
    std::string got;
    ASSERT_TRUE(CompareFormatReal("(EN10.1)", value, expect, got))
        << "Failed to format EN10.1, expected '" << expect << "', got '" << got
        << "'";
  }
}

TEST(IOApiTests, FormatIntegerValues) {
  using IntTestCaseTy = std::tuple<const char *, std::int64_t, const char *>;
  static const std::vector<IntTestCaseTy> intTestCases{
      {"(I4)", 0, "   0"},
      {"(I4)", 1, "   1"},
      {"(I4)", 9999, "9999"},
      {"(SP,I4)", 1, "  +1"},
      {"(SP,I4)", 9999, "****"},
      {"(SP,I4)", 999, "+999"},
      {"(I4)", -1, "  -1"},
      {"(I4)", -9999, "****"},
      {"(I4)", -999, "-999"},
      {"(I4.2)", 1, "  01"},
      {"(I4.2)", -1, " -01"},
      {"(I4.2)", 999, " 999"},
      {"(I4.4)", 999, "0999"},
      {"(I0)", 0, "0"},
      {"(I0)", 1, "1"},
      {"(I0)", 9999, "9999"},
      {"(SP,I0)", 1, "+1"},
      {"(SP,I0)", 9999, "+9999"},
      {"(SP,I0)", 999, "+999"},
      {"(I0)", -1, "-1"},
      {"(I0)", -9999, "-9999"},
      {"(I0)", -999, "-999"},
      {"(I0.2)", 1, "01"},
      {"(I0.2)", -1, "-01"},
      {"(I0.2)", 999, "999"},
      {"(I0.4)", 999, "0999"},
      {"(G4)", 0, "   0"},
      {"(G4)", 1, "   1"},
      {"(G4)", 9999, "9999"},
      {"(SP,G4)", 1, "  +1"},
      {"(SP,G4)", 9999, "****"},
      {"(SP,G4)", 999, "+999"},
      {"(G4)", -1, "  -1"},
      {"(G4)", -9999, "****"},
      {"(G4)", -999, "-999"},
      {"(G4.2)", 1, "   1"},
      {"(G4.2)", -1, "  -1"},
      {"(G4.2)", 999, " 999"},
      {"(G4.4)", 999, " 999"},
      {"(G0)", 0, "0"},
      {"(G0)", 1, "1"},
      {"(G0)", 9999, "9999"},
      {"(SP,G0)", 1, "+1"},
      {"(SP,G0)", 9999, "+9999"},
      {"(SP,G0)", 999, "+999"},
      {"(G0)", -1, "-1"},
      {"(G0)", -9999, "-9999"},
      {"(G0)", -999, "-999"},
      {"(G0.2)", 1, "1"},
      {"(G0.2)", -1, "-1"},
      {"(G0.2)", 999, "999"},
      {"(G0.4)", 999, "999"},
  };

  for (auto const &[fmt, value, expect] : intTestCases) {
    std::string got;
    ASSERT_TRUE(CompareFormatInteger(fmt, value, expect, got))
        << "Failed to format " << fmt << ", expected '" << expect << "', got '"
        << got << "'";
  }
}

//------------------------------------------------------------------------------
/// Tests for input formatting real values
//------------------------------------------------------------------------------

// Ensure double input values correctly map to raw uint64 values
TEST(IOApiTests, FormatDoubleInputValues) {
  using TestCaseTy = std::tuple<const char *, const char *, std::uint64_t>;
  static const std::vector<TestCaseTy> testCases{
      {"(F18.0)", "                 0", 0x0},
      {"(F18.0)", "                  ", 0x0},
      {"(F18.0)", "                -0", 0x8000000000000000},
      {"(F18.0)", "                01", 0x3ff0000000000000},
      {"(F18.0)", "                 1", 0x3ff0000000000000},
      {"(F18.0)", "              125.", 0x405f400000000000},
      {"(F18.0)", "              12.5", 0x4029000000000000},
      {"(F18.0)", "              1.25", 0x3ff4000000000000},
      {"(F18.0)", "             01.25", 0x3ff4000000000000},
      {"(F18.0)", "              .125", 0x3fc0000000000000},
      {"(F18.0)", "             0.125", 0x3fc0000000000000},
      {"(F18.0)", "             .0625", 0x3fb0000000000000},
      {"(F18.0)", "            0.0625", 0x3fb0000000000000},
      {"(F18.0)", "               125", 0x405f400000000000},
      {"(F18.1)", "               125", 0x4029000000000000},
      {"(F18.2)", "               125", 0x3ff4000000000000},
      {"(F18.3)", "               125", 0x3fc0000000000000},
      {"(-1P,F18.0)", "               125", 0x4093880000000000}, // 1250
      {"(1P,F18.0)", "               125", 0x4029000000000000}, // 12.5
      {"(BZ,F18.0)", "              125 ", 0x4093880000000000}, // 1250
      {"(BZ,F18.0)", "       125 . e +1 ", 0x42a6bcc41e900000}, // 1.25e13
      {"(BZ,F18.0)", "           .      ", 0x0},
      {"(BZ,F18.0)", "           . e +1 ", 0x0},
      {"(DC,F18.0)", "              12,5", 0x4029000000000000},
  };
  for (auto const &[format, data, want] : testCases) {
    auto cookie{IONAME(BeginInternalFormattedInput)(
        data, std::strlen(data), format, std::strlen(format))};
    union {
      double x;
      std::uint64_t raw;
    } u;
    u.raw = 0;

    // Read buffer into union value
    IONAME(EnableHandlers)(cookie, true, true, true, true, true);
    IONAME(InputReal64)(cookie, u.x);

    static constexpr int bufferSize{65};
    char iomsg[bufferSize];
    std::memset(iomsg, '\0', bufferSize - 1);

    // Ensure no errors were encountered reading input buffer into union value
    IONAME(GetIoMsg)(cookie, iomsg, bufferSize - 1);
    auto status{IONAME(EndIoStatement)(cookie)};
    ASSERT_EQ(status, 0) << '\'' << format << "' failed reading '" << data
                         << "', status " << static_cast<int>(status)
                         << " iomsg '" << iomsg << "'";

    // Ensure raw uint64 value matches expected conversion from double
    ASSERT_EQ(u.raw, want) << '\'' << format << "' failed reading '" << data
                           << "', want " << want << ", got " << u.raw;
  }
}
