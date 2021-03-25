//===-- flang/unittests/RuntimeGTest/CharacterTest.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Basic sanity tests of CHARACTER API; exhaustive testing will be done
// in Fortran.

#include "../../runtime/character.h"
#include "gtest/gtest.h"
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;

TEST(CharacterTests, AppendAndPad) {
  static constexpr int limitMax{8};
  static char buffer[limitMax];
  static std::size_t offset{0};
  for (std::size_t limit{0}; limit < limitMax; ++limit, offset = 0) {
    std::memset(buffer, 0, sizeof buffer);

    // Ensure appending characters does not overrun the limit
    offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "abc", 3);
    offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "DE", 2);
    ASSERT_LE(offset, limit) << "offset " << offset << ">" << limit;

    // Ensure whitespace padding does not overrun limit, the string is still
    // null-terminated, and string matches the expected value up to the limit.
    RTNAME(CharacterPad1)(buffer, limit, offset);
    EXPECT_EQ(buffer[limit], '\0')
        << "buffer[" << limit << "]='" << buffer[limit] << "'";
    buffer[limit] = buffer[limit] ? '\0' : buffer[limit];
    ASSERT_EQ(std::memcmp(buffer, "abcDE   ", limit), 0)
        << "buffer = '" << buffer << "'";
  }
}

TEST(CharacterTests, CharacterAppend1Overrun) {
  static constexpr int bufferSize{4};
  static constexpr std::size_t limit{2};
  static char buffer[bufferSize];
  static std::size_t offset{0};
  std::memset(buffer, 0, sizeof buffer);
  offset = RTNAME(CharacterAppend1)(buffer, limit, offset, "1234", bufferSize);
  ASSERT_EQ(offset, limit) << "CharacterAppend1 did not halt at limit = "
                           << limit << ", but at offset = " << offset;
}

//------------------------------------------------------------------------------
/// Tests and infrastructure for character comparison functions
//------------------------------------------------------------------------------

template <typename CHAR>
using ComparisonFuncTy =
    std::function<int(const CHAR *, const CHAR *, std::size_t, std::size_t)>;

using ComparisonFuncsTy = std::tuple<ComparisonFuncTy<char>,
    ComparisonFuncTy<char16_t>, ComparisonFuncTy<char32_t>>;

// These comparison functions are the systems under test in the
// CharacterComparisonTests test cases.
static ComparisonFuncsTy comparisonFuncs{
    RTNAME(CharacterCompareScalar1),
    RTNAME(CharacterCompareScalar2),
    RTNAME(CharacterCompareScalar4),
};

// Types of _values_ over which comparison tests are parameterized
template <typename CHAR>
using ComparisonParametersTy =
    std::vector<std::tuple<const CHAR *, const CHAR *, int, int, int>>;

using ComparisonTestCasesTy = std::tuple<ComparisonParametersTy<char>,
    ComparisonParametersTy<char16_t>, ComparisonParametersTy<char32_t>>;

static ComparisonTestCasesTy comparisonTestCases{
    {
        std::make_tuple("abc", "abc", 3, 3, 0),
        std::make_tuple("abc", "def", 3, 3, -1),
        std::make_tuple("ab ", "abc", 3, 2, 0),
        std::make_tuple("abc", "abc", 2, 3, -1),
    },
    {
        std::make_tuple(u"abc", u"abc", 3, 3, 0),
        std::make_tuple(u"abc", u"def", 3, 3, -1),
        std::make_tuple(u"ab ", u"abc", 3, 2, 0),
        std::make_tuple(u"abc", u"abc", 2, 3, -1),
    },
    {
        std::make_tuple(U"abc", U"abc", 3, 3, 0),
        std::make_tuple(U"abc", U"def", 3, 3, -1),
        std::make_tuple(U"ab ", U"abc", 3, 2, 0),
        std::make_tuple(U"abc", U"abc", 2, 3, -1),
    }};

template <typename CHAR>
struct CharacterComparisonTests : public ::testing::Test {
  CharacterComparisonTests()
      : parameters{std::get<ComparisonParametersTy<CHAR>>(comparisonTestCases)},
        characterComparisonFunc{
            std::get<ComparisonFuncTy<CHAR>>(comparisonFuncs)} {}
  ComparisonParametersTy<CHAR> parameters;
  ComparisonFuncTy<CHAR> characterComparisonFunc;
};

using CharacterTypes = ::testing::Types<char, char16_t, char32_t>;
TYPED_TEST_CASE(CharacterComparisonTests, CharacterTypes);

TYPED_TEST(CharacterComparisonTests, CompareCharacters) {
  for (auto &[x, y, xBytes, yBytes, expect] : this->parameters) {
    int cmp{this->characterComparisonFunc(x, y, xBytes, yBytes)};
    TypeParam buf[2][8];
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    ASSERT_EQ(cmp, expect) << "compare '" << x << "'(" << xBytes << ") to '"
                           << y << "'(" << yBytes << "), got " << cmp
                           << ", should be " << expect << '\n';

    // Perform the same test with the parameters reversed and the difference
    // negated
    std::swap(x, y);
    std::swap(xBytes, yBytes);
    expect = -expect;

    cmp = this->characterComparisonFunc(x, y, xBytes, yBytes);
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    ASSERT_EQ(cmp, expect) << "compare '" << x << "'(" << xBytes << ") to '"
                           << y << "'(" << yBytes << "), got " << cmp
                           << ", should be " << expect << '\n';
  }
}

// Test search functions INDEX(), SCAN(), and VERIFY()

template <typename CHAR>
using SearchFunction = std::function<std::size_t(
    const CHAR *, std::size_t, const CHAR *, std::size_t, bool)>;
template <template <typename> class FUNC>
using CharTypedFunctions =
    std::tuple<FUNC<char>, FUNC<char16_t>, FUNC<char32_t>>;
using SearchFunctions = CharTypedFunctions<SearchFunction>;
struct SearchTestCase {
  const char *x, *y;
  bool back;
  std::size_t expect;
};

template <typename CHAR>
void RunSearchTests(const char *which,
    const std::vector<SearchTestCase> &testCases,
    const SearchFunction<CHAR> &function) {
  for (const auto &t : testCases) {
    // Convert default character to desired kind
    std::size_t xLen{std::strlen(t.x)}, yLen{std::strlen(t.y)};
    std::basic_string<CHAR> x{t.x, t.x + xLen};
    std::basic_string<CHAR> y{t.y, t.y + yLen};
    auto got{function(x.data(), xLen, y.data(), yLen, t.back)};
    ASSERT_EQ(got, t.expect)
        << which << "('" << t.x << "','" << t.y << "',back=" << t.back
        << ") for CHARACTER(kind=" << sizeof(CHAR) << "): got " << got
        << ", expected " << t.expect;
  }
}

template <typename CHAR> struct SearchTests : public ::testing::Test {};
TYPED_TEST_CASE(SearchTests, CharacterTypes);

TYPED_TEST(SearchTests, IndexTests) {
  static SearchFunctions functions{
      RTNAME(Index1), RTNAME(Index2), RTNAME(Index4)};
  static std::vector<SearchTestCase> tests{
      {"", "", false, 1},
      {"", "", true, 1},
      {"a", "", false, 1},
      {"a", "", true, 2},
      {"", "a", false, 0},
      {"", "a", true, 0},
      {"aa", "a", false, 1},
      {"aa", "a", true, 2},
      {"Fortran that I ran", "that I ran", false, 9},
      {"Fortran that I ran", "that I ran", true, 9},
      {"Fortran that you ran", "that I ran", false, 0},
      {"Fortran that you ran", "that I ran", true, 0},
  };
  RunSearchTests(
      "INDEX", tests, std::get<SearchFunction<TypeParam>>(functions));
}

TYPED_TEST(SearchTests, ScanTests) {
  static SearchFunctions functions{RTNAME(Scan1), RTNAME(Scan2), RTNAME(Scan4)};
  static std::vector<SearchTestCase> tests{
      {"abc", "abc", false, 1},
      {"abc", "abc", true, 3},
      {"abc", "cde", false, 3},
      {"abc", "cde", true, 3},
      {"abc", "x", false, 0},
      {"", "x", false, 0},
  };
  RunSearchTests("SCAN", tests, std::get<SearchFunction<TypeParam>>(functions));
}

TYPED_TEST(SearchTests, VerifyTests) {
  static SearchFunctions functions{
      RTNAME(Verify1), RTNAME(Verify2), RTNAME(Verify4)};
  static std::vector<SearchTestCase> tests{
      {"abc", "abc", false, 0},
      {"abc", "abc", true, 0},
      {"abc", "cde", false, 1},
      {"abc", "cde", true, 2},
      {"abc", "x", false, 1},
      {"", "x", false, 0},
  };
  RunSearchTests(
      "VERIFY", tests, std::get<SearchFunction<TypeParam>>(functions));
}
