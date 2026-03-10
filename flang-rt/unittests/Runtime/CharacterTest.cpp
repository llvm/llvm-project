//===-- unittests/Runtime/CharacterTest.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Basic sanity tests of CHARACTER API; exhaustive testing will be done
// in Fortran.

#include "flang/Runtime/character.h"
#include "gtest/gtest.h"
#include "flang-rt/runtime/descriptor.h"
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

using namespace Fortran::runtime;

using CharacterTypes = ::testing::Types<char, char16_t, char32_t>;

// Helper for creating, allocating and filling up a descriptor with data from
// raw character literals, converted to the CHAR type used by the test.
template <typename CHAR>
OwningPtr<Descriptor> CreateDescriptor(const std::vector<SubscriptValue> &shape,
    const std::vector<const char *> &raw_strings) {
  std::size_t length{std::strlen(raw_strings[0])};

  OwningPtr<Descriptor> descriptor{Descriptor::Create(sizeof(CHAR), length,
      nullptr, shape.size(), nullptr, CFI_attribute_allocatable)};
  int rank{static_cast<int>(shape.size())};
  // Use a weird lower bound of 2 to flush out subscripting bugs
  for (int j{0}; j < rank; ++j) {
    descriptor->GetDimension(j).SetBounds(2, shape[j] + 1);
  }
  if (descriptor->Allocate(kNoAsyncObject) != 0) {
    return nullptr;
  }

  std::size_t offset = 0;
  for (const char *raw : raw_strings) {
    std::basic_string<CHAR> converted{raw, raw + length};
    std::copy(converted.begin(), converted.end(),
        descriptor->OffsetElement<CHAR>(offset * length * sizeof(CHAR)));
    ++offset;
  }

  return descriptor;
}

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

// Test ADJUSTL() and ADJUSTR()
template <typename CHAR> struct AdjustLRTests : public ::testing::Test {};
TYPED_TEST_SUITE(AdjustLRTests, CharacterTypes, );

struct AdjustLRTestCase {
  const char *input, *output;
};

template <typename CHAR>
void RunAdjustLRTest(const char *which,
    const std::function<void(
        Descriptor &, const Descriptor &, const char *, int)> &adjust,
    const char *inputRaw, const char *outputRaw) {
  OwningPtr<Descriptor> input{CreateDescriptor<CHAR>({}, {inputRaw})};
  ASSERT_NE(input, nullptr);
  ASSERT_TRUE(input->IsAllocated());

  StaticDescriptor<1> outputStaticDescriptor;
  Descriptor &output{outputStaticDescriptor.descriptor()};

  adjust(output, *input, /*sourceFile=*/nullptr, /*sourceLine=*/0);
  std::basic_string<CHAR> got{
      output.OffsetElement<CHAR>(), std::strlen(inputRaw)};
  std::basic_string<CHAR> expect{outputRaw, outputRaw + std::strlen(outputRaw)};
  ASSERT_EQ(got, expect) << which << "('" << inputRaw
                         << "') for CHARACTER(kind=" << sizeof(CHAR) << ")";
}

TYPED_TEST(AdjustLRTests, AdjustL) {
  static std::vector<AdjustLRTestCase> testcases{
      {"     where should the spaces be?", "where should the spaces be?     "},
      {"   leading and trailing whitespaces   ",
          "leading and trailing whitespaces      "},
      {"shouldn't change", "shouldn't change"},
  };

  for (const auto &t : testcases) {
    RunAdjustLRTest<TypeParam>("Adjustl", RTNAME(Adjustl), t.input, t.output);
  }
}

TYPED_TEST(AdjustLRTests, AdjustR) {
  static std::vector<AdjustLRTestCase> testcases{
      {"where should the spaces be?   ", "   where should the spaces be?"},
      {" leading and trailing whitespaces ",
          "  leading and trailing whitespaces"},
      {"shouldn't change", "shouldn't change"},
  };

  for (const auto &t : testcases) {
    RunAdjustLRTest<TypeParam>("Adjustr", RTNAME(Adjustr), t.input, t.output);
  }
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
        std::make_tuple("ab\xff", "ab ", 3, 2, 1),
        std::make_tuple("ab ", "ab\xff", 2, 3, -1),
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

TYPED_TEST_SUITE(CharacterComparisonTests, CharacterTypes, );

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
                           << y << "'(" << yBytes << "'), got " << cmp
                           << ", should be " << expect << '\n';
  }
}

// Test MIN() and MAX()
struct ExtremumTestCase {
  std::vector<SubscriptValue> shape; // Empty = scalar, non-empty = array.
  std::vector<const char *> x, y, expect;
};

template <typename CHAR>
void RunExtremumTests(const char *which,
    std::function<void(Descriptor &, const Descriptor &, const char *, int)>
        function,
    const std::vector<ExtremumTestCase> &testCases) {
  std::stringstream traceMessage;
  traceMessage << which << " for CHARACTER(kind=" << sizeof(CHAR) << ")";
  SCOPED_TRACE(traceMessage.str());

  for (const auto &t : testCases) {
    OwningPtr<Descriptor> x = CreateDescriptor<CHAR>(t.shape, t.x);
    OwningPtr<Descriptor> y = CreateDescriptor<CHAR>(t.shape, t.y);

    ASSERT_NE(x, nullptr);
    ASSERT_TRUE(x->IsAllocated());
    ASSERT_NE(y, nullptr);
    ASSERT_TRUE(y->IsAllocated());
    function(*x, *y, __FILE__, __LINE__);

    std::size_t length = x->ElementBytes() / sizeof(CHAR);
    for (std::size_t i = 0; i < t.x.size(); ++i) {
      std::basic_string<CHAR> got{
          x->OffsetElement<CHAR>(i * x->ElementBytes()), length};
      std::basic_string<CHAR> expect{
          t.expect[i], t.expect[i] + std::strlen(t.expect[i])};
      EXPECT_EQ(expect, got) << "inputs: '" << t.x[i] << "','" << t.y[i] << "'";
    }

    x->Deallocate();
    y->Deallocate();
  }
}

template <typename CHAR> struct ExtremumTests : public ::testing::Test {};
TYPED_TEST_SUITE(ExtremumTests, CharacterTypes, );

TYPED_TEST(ExtremumTests, MinTests) {
  static std::vector<ExtremumTestCase> tests{{{}, {"a"}, {"z"}, {"a"}},
      {{1}, {"zaaa"}, {"aa"}, {"aa  "}},
      {{1, 1}, {"aaz"}, {"aaaaa"}, {"aaaaa"}},
      {{2, 3}, {"a", "b", "c", "d", "E", "f"},
          {"xa", "ya", "az", "dd", "Sz", "cc"},
          {"a ", "b ", "az", "d ", "E ", "cc"}}};
  RunExtremumTests<TypeParam>("MIN", RTNAME(CharacterMin), tests);
}

TYPED_TEST(ExtremumTests, MaxTests) {
  static std::vector<ExtremumTestCase> tests{
      {{}, {"a"}, {"z"}, {"z"}},
      {{1}, {"zaa"}, {"aaaaa"}, {"zaa  "}},
      {{1, 1, 1}, {"aaaaa"}, {"aazaa"}, {"aazaa"}},
  };
  RunExtremumTests<TypeParam>("MAX", RTNAME(CharacterMax), tests);
}

template <typename CHAR>
void RunAllocationTest(const char *xRaw, const char *yRaw) {
  OwningPtr<Descriptor> x = CreateDescriptor<CHAR>({}, {xRaw});
  OwningPtr<Descriptor> y = CreateDescriptor<CHAR>({}, {yRaw});

  ASSERT_NE(x, nullptr);
  ASSERT_TRUE(x->IsAllocated());
  ASSERT_NE(y, nullptr);
  ASSERT_TRUE(y->IsAllocated());

  void *old = x->raw().base_addr;
  RTNAME(CharacterMin)(*x, *y, __FILE__, __LINE__);
  EXPECT_EQ(old, x->raw().base_addr);
}

TYPED_TEST(ExtremumTests, NoReallocate) {
  // Test that we don't reallocate if the accumulator is already large enough.
  RunAllocationTest<TypeParam>("loooooong", "short");
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
TYPED_TEST_SUITE(SearchTests, CharacterTypes, );

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
      {"aAA", "A", false, 2},
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

// Test REPEAT()
template <typename CHAR> struct RepeatTests : public ::testing::Test {};
TYPED_TEST_SUITE(RepeatTests, CharacterTypes, );

struct RepeatTestCase {
  std::size_t ncopies;
  const char *input, *output;
};

template <typename CHAR>
void RunRepeatTest(
    std::size_t ncopies, const char *inputRaw, const char *outputRaw) {
  OwningPtr<Descriptor> input{CreateDescriptor<CHAR>({}, {inputRaw})};
  ASSERT_NE(input, nullptr);
  ASSERT_TRUE(input->IsAllocated());

  StaticDescriptor<1> outputStaticDescriptor;
  Descriptor &output{outputStaticDescriptor.descriptor()};

  RTNAME(Repeat)(output, *input, ncopies);
  std::basic_string<CHAR> got{
      output.OffsetElement<CHAR>(), output.ElementBytes() / sizeof(CHAR)};
  std::basic_string<CHAR> expect{outputRaw, outputRaw + std::strlen(outputRaw)};
  ASSERT_EQ(got, expect) << "'" << inputRaw << "' * " << ncopies
                         << "' for CHARACTER(kind=" << sizeof(CHAR) << ")";
}

TYPED_TEST(RepeatTests, Repeat) {
  static std::vector<RepeatTestCase> testcases{
      {1, "just one copy", "just one copy"},
      {5, "copy.", "copy.copy.copy.copy.copy."},
      {0, "no copies", ""},
  };

  for (const auto &t : testcases) {
    RunRepeatTest<TypeParam>(t.ncopies, t.input, t.output);
  }
}

// Test TOKENIZE() - Form 1 and Form 2
// Helper to create a scalar character descriptor from a raw C string.
template <typename CHAR>
OwningPtr<Descriptor> CreateScalarDescriptor(const char *raw) {
  std::size_t len{std::strlen(raw)};
  OwningPtr<Descriptor> desc{Descriptor::Create(
      sizeof(CHAR), len, nullptr, 0, nullptr, CFI_attribute_other)};
  if (desc->Allocate(kNoAsyncObject) != 0) {
    return nullptr;
  }
  std::basic_string<CHAR> converted{raw, raw + len};
  std::copy(converted.begin(), converted.end(), desc->OffsetElement<CHAR>(0));
  return desc;
}

// Helper to create an unallocated allocatable character descriptor (rank 1,
// deferred length) for TOKENS or SEPARATOR output.
template <typename CHAR> StaticDescriptor<1> CreateAllocatableCharDescriptor() {
  StaticDescriptor<1> staticDesc;
  Descriptor &desc{staticDesc.descriptor()};
  desc.Establish(static_cast<int>(sizeof(CHAR)), static_cast<SubscriptValue>(0),
      nullptr, 1, nullptr, CFI_attribute_allocatable);
  desc.GetDimension(0).SetBounds(1, 0);
  return staticDesc;
}

// Helper to create an unallocated allocatable integer descriptor (rank 1)
// for FIRST or LAST output.
static StaticDescriptor<1> CreateAllocatableIntDescriptor() {
  StaticDescriptor<1> staticDesc;
  Descriptor &desc{staticDesc.descriptor()};
  desc.Establish(
      TypeCategory::Integer, 4, nullptr, 1, nullptr, CFI_attribute_allocatable);
  desc.GetDimension(0).SetBounds(1, 0);
  return staticDesc;
}

// Helper to extract a token string from the TOKENS descriptor.
template <typename CHAR>
std::basic_string<CHAR> GetToken(Descriptor &tokens, std::size_t index) {
  std::size_t elemBytes{tokens.ElementBytes()};
  std::size_t charLen{elemBytes / sizeof(CHAR)};
  const CHAR *data{tokens.OffsetElement<CHAR>(index * elemBytes)};
  return std::basic_string<CHAR>(data, charLen);
}

template <typename CHAR> struct TokenizeTests : public ::testing::Test {};
TYPED_TEST_SUITE(TokenizeTests, CharacterTypes, );

// Form 1: basic tokenization
TYPED_TEST(TokenizeTests, Form1Basic) {
  auto string{CreateScalarDescriptor<TypeParam>("first,second,third")};
  auto set{CreateScalarDescriptor<TypeParam>(",")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto tokensStatic{CreateAllocatableCharDescriptor<TypeParam>()};
  Descriptor &tokens{tokensStatic.descriptor()};

  RTNAME(Tokenize)(tokens, nullptr, *string, *set);

  // Expect 3 tokens: "first", "second", "third"
  ASSERT_TRUE(tokens.IsAllocated());
  EXPECT_EQ(tokens.GetDimension(0).Extent(), 3);
  // Longest token is "second" (6 chars)
  EXPECT_EQ(tokens.ElementBytes(), 6u * sizeof(TypeParam));

  // Tokens are blank-padded to max length
  std::basic_string<TypeParam> t0{GetToken<TypeParam>(tokens, 0)};
  std::basic_string<TypeParam> t1{GetToken<TypeParam>(tokens, 1)};
  std::basic_string<TypeParam> t2{GetToken<TypeParam>(tokens, 2)};
  std::basic_string<TypeParam> e0{'f', 'i', 'r', 's', 't', ' '};
  std::basic_string<TypeParam> e1{'s', 'e', 'c', 'o', 'n', 'd'};
  std::basic_string<TypeParam> e2{'t', 'h', 'i', 'r', 'd', ' '};
  EXPECT_EQ(t0, e0);
  EXPECT_EQ(t1, e1);
  EXPECT_EQ(t2, e2);
  tokens.Deallocate();
}

// Form 1: empty string produces one zero-length token
TYPED_TEST(TokenizeTests, Form1EmptyString) {
  auto string{CreateScalarDescriptor<TypeParam>("")};
  auto set{CreateScalarDescriptor<TypeParam>(",")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto tokensStatic{CreateAllocatableCharDescriptor<TypeParam>()};
  Descriptor &tokens{tokensStatic.descriptor()};

  RTNAME(Tokenize)(tokens, nullptr, *string, *set);

  ASSERT_TRUE(tokens.IsAllocated());
  EXPECT_EQ(tokens.GetDimension(0).Extent(), 1) << "empty string = 1 token";
  EXPECT_EQ(tokens.ElementBytes(), 0u) << "token length should be 0";
  EXPECT_EQ(tokens.GetDimension(0).LowerBound(), 1);
  tokens.Deallocate();
}

// Form 1: consecutive delimiters produce empty tokens
TYPED_TEST(TokenizeTests, Form1ConsecutiveDelimiters) {
  auto string{CreateScalarDescriptor<TypeParam>("a,,b")};
  auto set{CreateScalarDescriptor<TypeParam>(",")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto tokensStatic{CreateAllocatableCharDescriptor<TypeParam>()};
  Descriptor &tokens{tokensStatic.descriptor()};

  RTNAME(Tokenize)(tokens, nullptr, *string, *set);

  ASSERT_TRUE(tokens.IsAllocated());
  // Expect 3 tokens: "a", "", "b"
  EXPECT_EQ(tokens.GetDimension(0).Extent(), 3);
  tokens.Deallocate();
}

// Form 1: with SEPARATOR output
TYPED_TEST(TokenizeTests, Form1WithSeparator) {
  auto string{CreateScalarDescriptor<TypeParam>("a,b;c")};
  auto set{CreateScalarDescriptor<TypeParam>(",;")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto tokensStatic{CreateAllocatableCharDescriptor<TypeParam>()};
  Descriptor &tokens{tokensStatic.descriptor()};
  auto sepStatic{CreateAllocatableCharDescriptor<TypeParam>()};
  Descriptor &separator{sepStatic.descriptor()};

  RTNAME(Tokenize)(tokens, &separator, *string, *set);

  // Expect 3 tokens: "a", "b", "c"
  ASSERT_TRUE(tokens.IsAllocated());
  EXPECT_EQ(tokens.GetDimension(0).Extent(), 3);
  ASSERT_TRUE(separator.IsAllocated());
  // Expect 2 separators: ',' then ';'
  EXPECT_EQ(separator.GetDimension(0).Extent(), 2);
  EXPECT_EQ(separator.ElementBytes(), sizeof(TypeParam));

  // Check separator values: ',' then ';'
  const TypeParam *sep0{separator.OffsetElement<TypeParam>(0)};
  const TypeParam *sep1{
      separator.OffsetElement<TypeParam>(separator.ElementBytes())};
  EXPECT_EQ(*sep0, static_cast<TypeParam>(','));
  EXPECT_EQ(*sep1, static_cast<TypeParam>(';'));
  tokens.Deallocate();
  separator.Deallocate();
}

// Form 2: basic position output
TYPED_TEST(TokenizeTests, Form2Basic) {
  // From the standard example: "first,second,,fourth"
  auto string{CreateScalarDescriptor<TypeParam>("first,second,,fourth")};
  auto set{CreateScalarDescriptor<TypeParam>(",;")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto firstStatic{CreateAllocatableIntDescriptor()};
  Descriptor &first{firstStatic.descriptor()};
  auto lastStatic{CreateAllocatableIntDescriptor()};
  Descriptor &last{lastStatic.descriptor()};

  RTNAME(TokenizePositions)(first, last, *string, *set);

  ASSERT_TRUE(first.IsAllocated());
  ASSERT_TRUE(last.IsAllocated());
  EXPECT_EQ(first.GetDimension(0).Extent(), 4);
  EXPECT_EQ(last.GetDimension(0).Extent(), 4);

  //  Expect: FIRST = [1, 7, 14, 15], LAST = [5, 12, 13, 20]
  EXPECT_EQ(*first.OffsetElement<std::int32_t>(0 * sizeof(std::int32_t)), 1);
  EXPECT_EQ(*first.OffsetElement<std::int32_t>(1 * sizeof(std::int32_t)), 7);
  EXPECT_EQ(*first.OffsetElement<std::int32_t>(2 * sizeof(std::int32_t)), 14);
  EXPECT_EQ(*first.OffsetElement<std::int32_t>(3 * sizeof(std::int32_t)), 15);
  EXPECT_EQ(*last.OffsetElement<std::int32_t>(0 * sizeof(std::int32_t)), 5);
  EXPECT_EQ(*last.OffsetElement<std::int32_t>(1 * sizeof(std::int32_t)), 12);
  EXPECT_EQ(*last.OffsetElement<std::int32_t>(2 * sizeof(std::int32_t)), 13);
  EXPECT_EQ(*last.OffsetElement<std::int32_t>(3 * sizeof(std::int32_t)), 20);
  first.Deallocate();
  last.Deallocate();
}

// Form 2: empty string produces one token with FIRST=1, LAST=0
TYPED_TEST(TokenizeTests, Form2EmptyString) {
  auto string{CreateScalarDescriptor<TypeParam>("")};
  auto set{CreateScalarDescriptor<TypeParam>(",")};
  ASSERT_NE(string, nullptr);
  ASSERT_NE(set, nullptr);

  auto firstStatic{CreateAllocatableIntDescriptor()};
  Descriptor &first{firstStatic.descriptor()};
  auto lastStatic{CreateAllocatableIntDescriptor()};
  Descriptor &last{lastStatic.descriptor()};

  RTNAME(TokenizePositions)(first, last, *string, *set);

  ASSERT_TRUE(first.IsAllocated());
  ASSERT_TRUE(last.IsAllocated());
  EXPECT_EQ(first.GetDimension(0).Extent(), 1) << "empty string = 1 token";
  EXPECT_EQ(last.GetDimension(0).Extent(), 1);

  // Expect FIRST(1)=1, LAST(1)=0
  EXPECT_EQ(*first.OffsetElement<std::int32_t>(0), 1) << "FIRST(1) = 1";
  EXPECT_EQ(*last.OffsetElement<std::int32_t>(0), 0) << "LAST(1) = 0";
  first.Deallocate();
  last.Deallocate();
}

// Test F_C_STRING()
TEST(CharacterTests, FCString) {
  // Test 1: Default behavior (trim trailing blanks)
  {
    static char buffer[11]; // "abc       " = 10 chars
    std::memset(buffer, ' ', 10);
    std::memcpy(buffer, "abc", 3);

    StaticDescriptor<0> inputStaticDescriptor;
    Descriptor &input{inputStaticDescriptor.descriptor()};
    input.Establish(TypeCode{CFI_type_char}, /*elemLen=*/10, buffer, 0, nullptr,
        CFI_attribute_pointer);

    OwningPtr<Descriptor> result{Descriptor::Create(TypeCode{CFI_type_char}, 1,
        nullptr, 0, nullptr, CFI_attribute_allocatable)};

    RTNAME(FCString)(*result, input, /*asis=*/false);

    EXPECT_EQ(result->ElementBytes(), std::size_t(4)); // "abc\0" = 4 bytes
    const char *data = result->OffsetElement<char>();
    EXPECT_EQ(std::string(data, 4), std::string("abc\0", 4));

    result->Destroy();
  }

  // Test 2: Keep trailing blanks (asis=true)
  {
    static char buffer[11];
    std::memset(buffer, ' ', 10);
    std::memcpy(buffer, "abc", 3);

    StaticDescriptor<0> inputStaticDescriptor;
    Descriptor &input{inputStaticDescriptor.descriptor()};
    input.Establish(TypeCode{CFI_type_char}, /*elemLen=*/10, buffer, 0, nullptr,
        CFI_attribute_pointer);

    OwningPtr<Descriptor> result{Descriptor::Create(TypeCode{CFI_type_char}, 1,
        nullptr, 0, nullptr, CFI_attribute_allocatable)};

    RTNAME(FCString)(*result, input, /*asis=*/true);

    EXPECT_EQ(
        result->ElementBytes(), std::size_t(11)); // "abc       \0" = 11 bytes
    const char *data = result->OffsetElement<char>();
    EXPECT_EQ(data[3], ' '); // Verify space preserved
    EXPECT_EQ(data[10], '\0'); // Verify null terminator

    result->Destroy();
  }

  // Test 3: All blanks, trimmed
  {
    static char buffer[11];
    std::memset(buffer, ' ', 10);

    StaticDescriptor<0> inputStaticDescriptor;
    Descriptor &input{inputStaticDescriptor.descriptor()};
    input.Establish(TypeCode{CFI_type_char}, /*elemLen=*/10, buffer, 0, nullptr,
        CFI_attribute_pointer);

    OwningPtr<Descriptor> result{Descriptor::Create(TypeCode{CFI_type_char}, 1,
        nullptr, 0, nullptr, CFI_attribute_allocatable)};

    RTNAME(FCString)(*result, input, /*asis=*/false);

    EXPECT_EQ(result->ElementBytes(), std::size_t(1)); // Just "\0"
    const char *data = result->OffsetElement<char>();
    EXPECT_EQ(data[0], '\0');

    result->Destroy();
  }

  // Test 4: No trailing blanks
  {
    static char buffer[11];
    std::memcpy(buffer, "hello", 5);

    StaticDescriptor<0> inputStaticDescriptor;
    Descriptor &input{inputStaticDescriptor.descriptor()};
    input.Establish(TypeCode{CFI_type_char}, /*elemLen=*/5, buffer, 0, nullptr,
        CFI_attribute_pointer);

    OwningPtr<Descriptor> result{Descriptor::Create(TypeCode{CFI_type_char}, 1,
        nullptr, 0, nullptr, CFI_attribute_allocatable)};

    RTNAME(FCString)(*result, input, /*asis=*/false);

    EXPECT_EQ(result->ElementBytes(), std::size_t(6)); // "hello\0"
    const char *data = result->OffsetElement<char>();
    EXPECT_EQ(std::string(data, 6), std::string("hello\0", 6));

    result->Destroy();
  }
}
