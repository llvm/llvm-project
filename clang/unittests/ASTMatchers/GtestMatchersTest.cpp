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

#define GTEST_PRED_FORMAT1_(pred_format, v1, on_failure) \
  GTEST_ASSERT_(pred_format(#v1, v1), on_failure)

#define EXPECT_PRED_FORMAT1(pred_format, v1) \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT1(pred_format, v1) \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_FATAL_FAILURE_)

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

#define ASSERT_THAT(value, matcher) \
  ASSERT_PRED_FORMAT1(              \
      ::testing::internal::MakePredicateFormatterFromMatcher(matcher), value)
#define EXPECT_THAT(value, matcher) \
  EXPECT_PRED_FORMAT1(              \
      ::testing::internal::MakePredicateFormatterFromMatcher(matcher), value)

#define ASSERT_EQ(val1, val2) \
    ASSERT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, val1, val2)
#define ASSERT_NE(val1, val2) \
    ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)

#define GMOCK_ON_CALL_IMPL_(mock_expr, Setter, call)                    \
  ((mock_expr).gmock_##call)(::testing::internal::GetWithoutMatchers(), \
                             nullptr)                                   \
      .Setter(nullptr, 0, #mock_expr, #call)

#define ON_CALL(obj, call) \
  GMOCK_ON_CALL_IMPL_(obj, InternalDefaultActionSetAt, call)

#define EXPECT_CALL(obj, call) \
  GMOCK_ON_CALL_IMPL_(obj, InternalExpectedAt, call)

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

  // For implementing ASSERT_THAT() and EXPECT_THAT().  The template
  // argument M must be a type that can be converted to a matcher.
  template <typename M>
  class PredicateFormatterFromMatcher {
   public:
    explicit PredicateFormatterFromMatcher(M m) : matcher_(m) {}

    // This template () operator allows a PredicateFormatterFromMatcher
    // object to act as a predicate-formatter suitable for using with
    // Google Test's EXPECT_PRED_FORMAT1() macro.
    template <typename T>
    int operator()(const char* value_text, const T& x) const {
      return 0;
    }

   private:
    const M matcher_;
  };

  template <typename M>
  inline PredicateFormatterFromMatcher<M> MakePredicateFormatterFromMatcher(
      M matcher) {
    return PredicateFormatterFromMatcher<M>(matcher);
  }

  bool GetWithoutMatchers() { return false; }

  template <typename F>
  class MockSpec {
   public:
    MockSpec<F>() {}

    bool InternalDefaultActionSetAt(
        const char* file, int line, const char* obj, const char* call) {
      return false;
    }

    bool InternalExpectedAt(
        const char* file, int line, const char* obj, const char* call) {
      return false;
    }

    MockSpec<F> operator()(bool, void*) {
      return *this;
    }
  };  // class MockSpec

  }  // namespace internal

  template <typename T>
  int StrEq(T val) {
    return 0;
  }
  template <typename T>
  int Eq(T val) {
    return 0;
  }

  }  // namespace testing

  class Mock {
    public:
    Mock() {}
    testing::internal::MockSpec<int> gmock_TwoArgsMethod(int, int) {
      return testing::internal::MockSpec<int>();
    }
    testing::internal::MockSpec<int> gmock_TwoArgsMethod(bool, void*) {
      return testing::internal::MockSpec<int>();
    }
  };  // class Mock
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

TEST(GtestExpectTest, ThatShouldMatchAssertThat) {
  std::string Input = R"cc(
    using ::testing::Eq;
    void Test() { ASSERT_THAT(2, Eq(2)); }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestAssertThat(
          expr(), callExpr(callee(functionDecl(hasName("::testing::Eq")))))));
}

TEST(GtestExpectTest, ThatShouldMatchExpectThat) {
  std::string Input = R"cc(
    using ::testing::Eq;
    void Test() { EXPECT_THAT(2, Eq(2)); }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestExpectThat(
          expr(), callExpr(callee(functionDecl(hasName("::testing::Eq")))))));
}

TEST(GtestOnCallTest, CallShouldMatchOnCallWithoutParams1) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      ON_CALL(mock, TwoArgsMethod);
    }
  )cc";
  EXPECT_TRUE(matches(wrapGtest(Input),
                      gtestOnCall(expr(hasType(cxxRecordDecl(hasName("Mock")))),
                                  "TwoArgsMethod", MockArgs::None)));
}

TEST(GtestOnCallTest, CallShouldMatchOnCallWithoutParams2) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      ON_CALL(mock, TwoArgsMethod);
    }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestOnCall(cxxMemberCallExpr(
                      callee(functionDecl(hasName("gmock_TwoArgsMethod"))))
                      .bind("mock_call"),
                  MockArgs::None)));
}

TEST(GtestOnCallTest, CallShouldMatchOnCallWithParams1) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      ON_CALL(mock, TwoArgsMethod(1, 2));
    }
  )cc";
  EXPECT_TRUE(matches(wrapGtest(Input),
                      gtestOnCall(expr(hasType(cxxRecordDecl(hasName("Mock")))),
                                  "TwoArgsMethod", MockArgs::Some)));
}

TEST(GtestOnCallTest, CallShouldMatchOnCallWithParams2) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      ON_CALL(mock, TwoArgsMethod(1, 2));
    }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestOnCall(cxxMemberCallExpr(
                      callee(functionDecl(hasName("gmock_TwoArgsMethod"))))
                      .bind("mock_call"),
                  MockArgs::Some)));
}

TEST(GtestExpectCallTest, CallShouldMatchExpectCallWithoutParams1) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      EXPECT_CALL(mock, TwoArgsMethod);
    }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input),
              gtestExpectCall(expr(hasType(cxxRecordDecl(hasName("Mock")))),
                              "TwoArgsMethod", MockArgs::None)));
}

TEST(GtestExpectCallTest, CallShouldMatchExpectCallWithoutParams2) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      EXPECT_CALL(mock, TwoArgsMethod);
    }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestExpectCall(cxxMemberCallExpr(
                          callee(functionDecl(hasName("gmock_TwoArgsMethod"))))
                          .bind("mock_call"),
                      MockArgs::None)));
}

TEST(GtestExpectCallTest, CallShouldMatchExpectCallWithParams1) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      EXPECT_CALL(mock, TwoArgsMethod(1, 2));
    }
  )cc";
  EXPECT_TRUE(
      matches(wrapGtest(Input),
              gtestExpectCall(expr(hasType(cxxRecordDecl(hasName("Mock")))),
                              "TwoArgsMethod", MockArgs::Some)));
}

TEST(GtestExpectCallTest, CallShouldMatchExpectCallWithParams2) {
  std::string Input = R"cc(
    void Test() {
      Mock mock;
      EXPECT_CALL(mock, TwoArgsMethod(1, 2));
    }
  )cc";
  EXPECT_TRUE(matches(
      wrapGtest(Input),
      gtestExpectCall(cxxMemberCallExpr(
                          callee(functionDecl(hasName("gmock_TwoArgsMethod"))))
                          .bind("mock_call"),
                      MockArgs::Some)));
}

} // end namespace ast_matchers
} // end namespace clang
