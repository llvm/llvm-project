//===- unittests/ASTMatchers/GTestMatchersTest.cpp - GTest matcher unit tests //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/GtestMatchers.h"

namespace clang {
namespace ast_matchers {

constexpr llvm::StringLiteral GtestMockDecls = R"cc(
  static int testerr;

#define GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    switch (0)                          \
    case 0:                             \
    default:  // NOLINT

#define GTEST_NONFATAL_FAILURE_(code) testerr = code

#define GTEST_FATAL_FAILURE_(code) testerr = code

#define GTEST_ASSERT_(expression, on_failure) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_               \
    if (const int gtest_ar = (expression))      \
      ;                                         \
    else                                        \
      on_failure(gtest_ar)

  // Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT2.
  // Don't use this in your code.
#define GTEST_PRED_FORMAT2_(pred_format, v1, v2, on_failure) \
    GTEST_ASSERT_(pred_format(#v1, #v2, v1, v2), on_failure)

#define ASSERT_PRED_FORMAT2(pred_format, v1, v2) \
    GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_FATAL_FAILURE_)
#define EXPECT_PRED_FORMAT2(pred_format, v1, v2) \
    GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_NONFATAL_FAILURE_)

#define EXPECT_EQ(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, val1, val2)
#define EXPECT_NE(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)
#define EXPECT_GE(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGE, val1, val2)
#define EXPECT_GT(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGT, val1, val2)
#define EXPECT_LE(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLE, val1, val2)
#define EXPECT_LT(val1, val2) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLT, val1, val2)

#define ASSERT_EQ(val1, val2) \
    ASSERT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, val1, val2)
#define ASSERT_NE(val1, val2) \
    ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)

  namespace testing {
  namespace internal {
  class EqHelper {
   public:
    // This templatized version is for the general case.
    template <typename T1, typename T2>
    static int Compare(const char* lhs_expression, const char* rhs_expression,
                       const T1& lhs, const T2& rhs) {
      return 0;
    }
  };
  template <typename T1, typename T2>
  int CmpHelperNE(const char* expr1, const char* expr2, const T1& val1,
                  const T2& val2) {
    return 0;
  }
  template <typename T1, typename T2>
  int CmpHelperGE(const char* expr1, const char* expr2, const T1& val1,
                  const T2& val2) {
    return 0;
  }
  template <typename T1, typename T2>
  int CmpHelperGT(const char* expr1, const char* expr2, const T1& val1,
                  const T2& val2) {
    return 0;
  }
  template <typename T1, typename T2>
  int CmpHelperLE(const char* expr1, const char* expr2, const T1& val1,
                  const T2& val2) {
    return 0;
  }
  template <typename T1, typename T2>
  int CmpHelperLT(const char* expr1, const char* expr2, const T1& val1,
                  const T2& val2) {
    return 0;
  }
  }  // namespace internal
  }  // namespace testing
)cc";

static std::string wrapGtest(llvm::StringRef Input) {
  return (GtestMockDecls + Input).str();
}

TEST(GtestAssertTest, ShouldMatchAssert) {
  std::string Input = R"cc(
    void Test() { ASSERT_EQ(1010, 4321); }
  )cc";
  EXPECT_TRUE(matches(wrapGtest(Input),
                      gtestAssert(GtestCmp::Eq, integerLiteral(equals(1010)),
                                  integerLiteral(equals(4321)))));
}

TEST(GtestAssertTest, ShouldNotMatchExpect) {
  std::string Input = R"cc(
    void Test() { EXPECT_EQ(2, 3); }
  )cc";
  EXPECT_TRUE(
      notMatches(wrapGtest(Input), gtestAssert(GtestCmp::Eq, expr(), expr())));
}

TEST(GtestAssertTest, ShouldMatchNestedAssert) {
  std::string Input = R"cc(
    #define WRAPPER(a, b) ASSERT_EQ(a, b)
    void Test() { WRAPPER(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestAssert(GtestCmp::Eq, expr(), expr())));
}

TEST(GtestExpectTest, ShouldMatchExpect) {
  std::string Input = R"cc(
    void Test() { EXPECT_EQ(1010, 4321); }
  )cc";
  EXPECT_TRUE(matches(wrapGtest(Input),
                      gtestExpect(GtestCmp::Eq, integerLiteral(equals(1010)),
                                  integerLiteral(equals(4321)))));
}

TEST(GtestExpectTest, ShouldNotMatchAssert) {
  std::string Input = R"cc(
    void Test() { ASSERT_EQ(2, 3); }
  )cc";
  EXPECT_TRUE(
      notMatches(wrapGtest(Input), gtestExpect(GtestCmp::Eq, expr(), expr())));
}

TEST(GtestExpectTest, NeShouldMatchExpectNe) {
  std::string Input = R"cc(
    void Test() { EXPECT_NE(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestExpect(GtestCmp::Ne, expr(), expr())));
}

TEST(GtestExpectTest, LeShouldMatchExpectLe) {
  std::string Input = R"cc(
    void Test() { EXPECT_LE(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestExpect(GtestCmp::Le, expr(), expr())));
}

TEST(GtestExpectTest, LtShouldMatchExpectLt) {
  std::string Input = R"cc(
    void Test() { EXPECT_LT(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestExpect(GtestCmp::Lt, expr(), expr())));
}

TEST(GtestExpectTest, GeShouldMatchExpectGe) {
  std::string Input = R"cc(
    void Test() { EXPECT_GE(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestExpect(GtestCmp::Ge, expr(), expr())));
}

TEST(GtestExpectTest, GtShouldMatchExpectGt) {
  std::string Input = R"cc(
    void Test() { EXPECT_GT(2, 3); }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input), gtestExpect(GtestCmp::Gt, expr(), expr())));
}

} // end namespace ast_matchers
} // end namespace clang
