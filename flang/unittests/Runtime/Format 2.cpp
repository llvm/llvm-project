//===-- flang/unittests/Runtime/Format.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "../runtime/connection.h"
#include "../runtime/format-implementation.h"
#include "../runtime/io-error.h"
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;
using namespace std::literals::string_literals;

using ResultsTy = std::vector<std::string>;

// A test harness context for testing FormatControl
class TestFormatContext : public IoErrorHandler {
public:
  using CharType = char;
  TestFormatContext() : IoErrorHandler{"format.cpp", 1} {}
  bool Emit(const char *, std::size_t, std::size_t = 0);
  bool AdvanceRecord(int = 1);
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);
  void Report(const std::optional<DataEdit> &);
  ResultsTy results;
  MutableModes &mutableModes() { return mutableModes_; }
  ConnectionState &GetConnectionState() { return connectionState_; }

private:
  MutableModes mutableModes_;
  ConnectionState connectionState_;
};

bool TestFormatContext::Emit(const char *s, std::size_t len, std::size_t) {
  std::string str{s, len};
  results.push_back("'"s + str + '\'');
  return true;
}

bool TestFormatContext::AdvanceRecord(int n) {
  while (n-- > 0) {
    results.emplace_back("/");
  }
  return true;
}

void TestFormatContext::HandleAbsolutePosition(std::int64_t n) {
  results.push_back("T"s + std::to_string(n));
}

void TestFormatContext::HandleRelativePosition(std::int64_t n) {
  if (n < 0) {
    results.push_back("TL"s + std::to_string(-n));
  } else {
    results.push_back(std::to_string(n) + 'X');
  }
}

void TestFormatContext::Report(const std::optional<DataEdit> &edit) {
  if (edit) {
    std::string str{edit->descriptor};
    if (edit->repeat != 1) {
      str = std::to_string(edit->repeat) + '*' + str;
    }
    if (edit->variation) {
      str += edit->variation;
    }
    if (edit->width) {
      str += std::to_string(*edit->width);
    }
    if (edit->digits) {
      str += "."s + std::to_string(*edit->digits);
    }
    if (edit->expoDigits) {
      str += "E"s + std::to_string(*edit->expoDigits);
    }
    // modes?
    results.push_back(str);
  } else {
    results.push_back("(nullopt)"s);
  }
}

struct FormatTests : public CrashHandlerFixture {};

TEST(FormatTests, FormatStringTraversal) {

  using ParamsTy = std::tuple<int, const char *, ResultsTy, int>;

  static const std::vector<ParamsTy> params{
      {1, "('PI=',F9.7)", ResultsTy{"'PI='", "F9.7"}, 1},
      {1, "(3HPI=F9.7)", ResultsTy{"'PI='", "F9.7"}, 1},
      {1, "(3HPI=/F9.7)", ResultsTy{"'PI='", "/", "F9.7"}, 1},
      {2, "('PI=',F9.7)", ResultsTy{"'PI='", "F9.7", "/", "'PI='", "F9.7"}, 1},
      {2, "(2('PI=',F9.7),'done')",
          ResultsTy{"'PI='", "F9.7", "'PI='", "F9.7", "'done'"}, 1},
      {2, "(3('PI=',F9.7,:),'tooFar')",
          ResultsTy{"'PI='", "F9.7", "'PI='", "F9.7"}, 1},
      {2, "(*('PI=',F9.7,:))", ResultsTy{"'PI='", "F9.7", "'PI='", "F9.7"}, 1},
      {1, "(3F9.7)", ResultsTy{"2*F9.7"}, 2},
      {9, "((I4,2(E10.1)))",
          ResultsTy{"I4", "E10.1", "E10.1", "/", "I4", "E10.1", "E10.1", "/",
              "I4", "E10.1", "E10.1"},
          1},
      {1, "(F)", ResultsTy{"F"}, 1}, // missing 'w'
  };

  for (const auto &[n, format, expect, repeat] : params) {
    TestFormatContext context;
    FormatControl<decltype(context)> control{
        context, format, std::strlen(format)};

    for (auto i{0}; i < n; i++) {
      context.Report(/*edit=*/control.GetNextDataEdit(context, repeat));
    }
    control.Finish(context);

    auto iostat{context.GetIoStat()};
    ASSERT_EQ(iostat, 0) << "Expected iostat == 0, but GetIoStat() == "
                         << iostat;

    // Create strings of the expected/actual results for printing errors
    std::string allExpectedResults{""}, allActualResults{""};
    for (const auto &res : context.results) {
      allActualResults += " "s + res;
    }
    for (const auto &res : expect) {
      allExpectedResults += " "s + res;
    }

    const auto &results = context.results;
    ASSERT_EQ(expect, results) << "Expected '" << allExpectedResults
                               << "' but got '" << allActualResults << "'";
  }
}

struct InvalidFormatFailure : CrashHandlerFixture {};

TEST(InvalidFormatFailure, ParenMismatch) {
  static constexpr const char *format{"("};
  static constexpr int repeat{1};

  TestFormatContext context;
  FormatControl<decltype(context)> control{
      context, format, std::strlen(format)};

  ASSERT_DEATH(
      context.Report(/*edit=*/control.GetNextDataEdit(context, repeat)),
      R"(FORMAT missing at least one '\)')");
}

TEST(InvalidFormatFailure, MissingPrecision) {
  static constexpr const char *format{"(F9.)"};
  static constexpr int repeat{1};

  TestFormatContext context;
  FormatControl<decltype(context)> control{
      context, format, std::strlen(format)};

  ASSERT_DEATH(
      context.Report(/*edit=*/control.GetNextDataEdit(context, repeat)),
      R"(Invalid FORMAT: integer expected at '\)')");
}

TEST(InvalidFormatFailure, MissingFormatWidthWithDigits) {
  static constexpr const char *format{"(F.9)"};
  static constexpr int repeat{1};

  TestFormatContext context;
  FormatControl<decltype(context)> control{
      context, format, std::strlen(format)};

  ASSERT_DEATH(
      context.Report(/*edit=*/control.GetNextDataEdit(context, repeat)),
      "Invalid FORMAT: integer expected at '.'");
}
