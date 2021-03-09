//===-- flang/unittests/RuntimeGTest/CharacterTest.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

//------------------------------------------------------------------------------
/// Tests and infrastructure for Scan functions
//------------------------------------------------------------------------------

template <typename CHAR>
using ScanFuncTy = std::function<int(
    const CHAR *, std::size_t, const CHAR *, std::size_t, bool)>;

using ScanFuncsTy =
    std::tuple<ScanFuncTy<char>, ScanFuncTy<char16_t>, ScanFuncTy<char32_t>>;

// These functions are the systems under test in CharacterScanTests test cases.
static ScanFuncsTy scanFuncs{
    RTNAME(Scan1),
    RTNAME(Scan2),
    RTNAME(Scan4),
};

// Types of _values_ over which tests are parameterized
template <typename CHAR>
using ScanParametersTy =
    std::vector<std::tuple<const CHAR *, const CHAR *, bool, int>>;

using ScanTestCasesTy = std::tuple<ScanParametersTy<char>,
    ScanParametersTy<char16_t>, ScanParametersTy<char32_t>>;

static ScanTestCasesTy scanTestCases{
    {
        std::make_tuple("abc", "abc", false, 1),
        std::make_tuple("abc", "abc", true, 3),
        std::make_tuple("abc", "cde", false, 3),
        std::make_tuple("abc", "cde", true, 3),
        std::make_tuple("abc", "x", false, 0),
        std::make_tuple("", "x", false, 0),
    },
    {
        std::make_tuple(u"abc", u"abc", false, 1),
        std::make_tuple(u"abc", u"abc", true, 3),
        std::make_tuple(u"abc", u"cde", false, 3),
        std::make_tuple(u"abc", u"cde", true, 3),
        std::make_tuple(u"abc", u"x", false, 0),
        std::make_tuple(u"", u"x", false, 0),
    },
    {
        std::make_tuple(U"abc", U"abc", false, 1),
        std::make_tuple(U"abc", U"abc", true, 3),
        std::make_tuple(U"abc", U"cde", false, 3),
        std::make_tuple(U"abc", U"cde", true, 3),
        std::make_tuple(U"abc", U"x", false, 0),
        std::make_tuple(U"", U"x", false, 0),
    }};

template <typename CHAR> struct CharacterScanTests : public ::testing::Test {
  CharacterScanTests()
      : parameters{std::get<ScanParametersTy<CHAR>>(scanTestCases)},
        characterScanFunc{std::get<ScanFuncTy<CHAR>>(scanFuncs)} {}
  ScanParametersTy<CHAR> parameters;
  ScanFuncTy<CHAR> characterScanFunc;
};

// Type-parameterized over the same character types as CharacterComparisonTests
TYPED_TEST_CASE(CharacterScanTests, CharacterTypes);

TYPED_TEST(CharacterScanTests, ScanCharacters) {
  for (auto const &[str, set, back, expect] : this->parameters) {
    auto res{
        this->characterScanFunc(str, std::char_traits<TypeParam>::length(str),
            set, std::char_traits<TypeParam>::length(set), back)};
    ASSERT_EQ(res, expect) << "Scan(" << str << ',' << set << ",back=" << back
                           << "): got " << res << ", should be " << expect;
  }
}

//------------------------------------------------------------------------------
/// Tests and infrastructure for Verify functions
//------------------------------------------------------------------------------
template <typename CHAR>
using VerifyFuncTy = std::function<int(
    const CHAR *, std::size_t, const CHAR *, std::size_t, bool)>;

using VerifyFuncsTy = std::tuple<VerifyFuncTy<char>, VerifyFuncTy<char16_t>,
    VerifyFuncTy<char32_t>>;

// These functions are the systems under test in CharacterVerifyTests test cases
static VerifyFuncsTy verifyFuncs{
    RTNAME(Verify1),
    RTNAME(Verify2),
    RTNAME(Verify4),
};

// Types of _values_ over which tests are parameterized
template <typename CHAR>
using VerifyParametersTy =
    std::vector<std::tuple<const CHAR *, const CHAR *, bool, int>>;

using VerifyTestCasesTy = std::tuple<VerifyParametersTy<char>,
    VerifyParametersTy<char16_t>, VerifyParametersTy<char32_t>>;

static VerifyTestCasesTy verifyTestCases{
    {
        std::make_tuple("abc", "abc", false, 0),
        std::make_tuple("abc", "abc", true, 0),
        std::make_tuple("abc", "cde", false, 1),
        std::make_tuple("abc", "cde", true, 2),
        std::make_tuple("abc", "x", false, 1),
        std::make_tuple("", "x", false, 0),
    },
    {
        std::make_tuple(u"abc", u"abc", false, 0),
        std::make_tuple(u"abc", u"abc", true, 0),
        std::make_tuple(u"abc", u"cde", false, 1),
        std::make_tuple(u"abc", u"cde", true, 2),
        std::make_tuple(u"abc", u"x", false, 1),
        std::make_tuple(u"", u"x", false, 0),
    },
    {
        std::make_tuple(U"abc", U"abc", false, 0),
        std::make_tuple(U"abc", U"abc", true, 0),
        std::make_tuple(U"abc", U"cde", false, 1),
        std::make_tuple(U"abc", U"cde", true, 2),
        std::make_tuple(U"abc", U"x", false, 1),
        std::make_tuple(U"", U"x", false, 0),
    }};

template <typename CHAR> struct CharacterVerifyTests : public ::testing::Test {
  CharacterVerifyTests()
      : parameters{std::get<VerifyParametersTy<CHAR>>(verifyTestCases)},
        characterVerifyFunc{std::get<VerifyFuncTy<CHAR>>(verifyFuncs)} {}
  VerifyParametersTy<CHAR> parameters;
  VerifyFuncTy<CHAR> characterVerifyFunc;
};

// Type-parameterized over the same character types as CharacterComparisonTests
TYPED_TEST_CASE(CharacterVerifyTests, CharacterTypes);

TYPED_TEST(CharacterVerifyTests, VerifyCharacters) {
  for (auto const &[str, set, back, expect] : this->parameters) {
    auto res{
        this->characterVerifyFunc(str, std::char_traits<TypeParam>::length(str),
            set, std::char_traits<TypeParam>::length(set), back)};
    ASSERT_EQ(res, expect) << "Verify(" << str << ',' << set << ",back=" << back
                           << "): got " << res << ", should be " << expect;
  }
}
