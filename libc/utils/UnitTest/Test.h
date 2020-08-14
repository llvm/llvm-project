//===-- Base class for libc unittests ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_H
#define LLVM_LIBC_UTILS_UNITTEST_H

// This file can only include headers from utils/CPP/ or utils/testutils. No
// other headers should be included.

#include "utils/CPP/TypeTraits.h"
#include "utils/testutils/ExecuteFunction.h"
#include "utils/testutils/StreamWrapper.h"

namespace __llvm_libc {
namespace testing {

class RunContext;

// Only the following conditions are supported. Notice that we do not have
// a TRUE or FALSE condition. That is because, C library funtions do not
// return boolean values, but use integral return values to indicate true or
// false conditions. Hence, it is more appropriate to use the other comparison
// conditions for such cases.
enum TestCondition {
  Cond_None,
  Cond_EQ,
  Cond_NE,
  Cond_LT,
  Cond_LE,
  Cond_GT,
  Cond_GE,
};

namespace internal {

template <typename ValType>
bool test(RunContext &Ctx, TestCondition Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, const char *File,
          unsigned long Line);

} // namespace internal

struct MatcherBase {
  virtual ~MatcherBase() {}
  virtual void explainError(testutils::StreamWrapper &OS) {
    OS << "unknown error\n";
  }
};

template <typename T> struct Matcher : public MatcherBase { bool match(T &t); };

// NOTE: One should not create instances and call methods on them directly. One
// should use the macros TEST or TEST_F to write test cases.
class Test {
private:
  Test *Next = nullptr;

public:
  virtual ~Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runTests();

protected:
  static void addTest(Test *T);

  // We make use of a template function, with |LHS| and |RHS| as explicit
  // parameters, for enhanced type checking. Other gtest like unittest
  // frameworks have a similar function which takes a boolean argument
  // instead of the explicit |LHS| and |RHS| arguments. This boolean argument
  // is the result of the |Cond| operation on |LHS| and |RHS|. Though not bad,
  // |Cond| on mismatched |LHS| and |RHS| types can potentially succeed because
  // of type promotion.
  template <typename ValType,
            cpp::EnableIfType<cpp::IsIntegral<ValType>::Value, int> = 0>
  static bool test(RunContext &Ctx, TestCondition Cond, ValType LHS,
                   ValType RHS, const char *LHSStr, const char *RHSStr,
                   const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, File, Line);
  }

  template <
      typename ValType,
      cpp::EnableIfType<cpp::IsPointerType<ValType>::Value, ValType> = nullptr>
  static bool test(RunContext &Ctx, TestCondition Cond, ValType LHS,
                   ValType RHS, const char *LHSStr, const char *RHSStr,
                   const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, (unsigned long long)LHS,
                          (unsigned long long)RHS, LHSStr, RHSStr, File, Line);
  }

  static bool testStrEq(RunContext &Ctx, const char *LHS, const char *RHS,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

  static bool testStrNe(RunContext &Ctx, const char *LHS, const char *RHS,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

  static bool testMatch(RunContext &Ctx, bool MatchResult, MatcherBase &Matcher,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

  static bool testProcessExits(RunContext &Ctx, testutils::FunctionCaller *Func,
                               int ExitCode, const char *LHSStr,
                               const char *RHSStr, const char *File,
                               unsigned long Line);

  static bool testProcessKilled(RunContext &Ctx,
                                testutils::FunctionCaller *Func, int Signal,
                                const char *LHSStr, const char *RHSStr,
                                const char *File, unsigned long Line);

  template <typename Func> testutils::FunctionCaller *createCallable(Func f) {
    struct Callable : public testutils::FunctionCaller {
      Func f;
      Callable(Func f) : f(f) {}
      void operator()() override { f(); }
    };

    return new Callable(f);
  }

private:
  virtual void Run(RunContext &Ctx) = 0;
  virtual const char *getName() const = 0;

  static Test *Start;
  static Test *End;
};

} // namespace testing
} // namespace __llvm_libc

#define TEST(SuiteName, TestName)                                              \
  class SuiteName##_##TestName : public __llvm_libc::testing::Test {           \
  public:                                                                      \
    SuiteName##_##TestName() { addTest(this); }                                \
    void Run(__llvm_libc::testing::RunContext &) override;                     \
    const char *getName() const override { return #SuiteName "." #TestName; }  \
  };                                                                           \
  SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
  void SuiteName##_##TestName::Run(__llvm_libc::testing::RunContext &Ctx)

#define TEST_F(SuiteClass, TestName)                                           \
  class SuiteClass##_##TestName : public SuiteClass {                          \
  public:                                                                      \
    SuiteClass##_##TestName() { addTest(this); }                               \
    void Run(__llvm_libc::testing::RunContext &) override;                     \
    const char *getName() const override { return #SuiteClass "." #TestName; } \
  };                                                                           \
  SuiteClass##_##TestName SuiteClass##_##TestName##_Instance;                  \
  void SuiteClass##_##TestName::Run(__llvm_libc::testing::RunContext &Ctx)

#define EXPECT_EQ(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_EQ, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_EQ(LHS, RHS)                                                    \
  if (!EXPECT_EQ(LHS, RHS))                                                    \
  return

#define EXPECT_NE(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_NE, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_NE(LHS, RHS)                                                    \
  if (!EXPECT_NE(LHS, RHS))                                                    \
  return

#define EXPECT_LT(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_LT, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_LT(LHS, RHS)                                                    \
  if (!EXPECT_LT(LHS, RHS))                                                    \
  return

#define EXPECT_LE(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_LE, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_LE(LHS, RHS)                                                    \
  if (!EXPECT_LE(LHS, RHS))                                                    \
  return

#define EXPECT_GT(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_GT, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_GT(LHS, RHS)                                                    \
  if (!EXPECT_GT(LHS, RHS))                                                    \
  return

#define EXPECT_GE(LHS, RHS)                                                    \
  __llvm_libc::testing::Test::test(Ctx, __llvm_libc::testing::Cond_GE, (LHS),  \
                                   (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_GE(LHS, RHS)                                                    \
  if (!EXPECT_GE(LHS, RHS))                                                    \
  return

#define EXPECT_STREQ(LHS, RHS)                                                 \
  __llvm_libc::testing::Test::testStrEq(Ctx, (LHS), (RHS), #LHS, #RHS,         \
                                        __FILE__, __LINE__)
#define ASSERT_STREQ(LHS, RHS)                                                 \
  if (!EXPECT_STREQ(LHS, RHS))                                                 \
  return

#define EXPECT_STRNE(LHS, RHS)                                                 \
  __llvm_libc::testing::Test::testStrNe(Ctx, (LHS), (RHS), #LHS, #RHS,         \
                                        __FILE__, __LINE__)
#define ASSERT_STRNE(LHS, RHS)                                                 \
  if (!EXPECT_STRNE(LHS, RHS))                                                 \
  return

#define EXPECT_TRUE(VAL) EXPECT_EQ((VAL), true)

#define ASSERT_TRUE(VAL)                                                       \
  if (!EXPECT_TRUE(VAL))                                                       \
  return

#define EXPECT_FALSE(VAL) EXPECT_EQ((VAL), false)

#define ASSERT_FALSE(VAL)                                                      \
  if (!EXPECT_FALSE(VAL))                                                      \
  return

#define EXPECT_EXITS(FUNC, EXIT)                                               \
  __llvm_libc::testing::Test::testProcessExits(                                \
      Ctx, __llvm_libc::testing::Test::createCallable(FUNC), EXIT, #FUNC,      \
      #EXIT, __FILE__, __LINE__)

#define ASSERT_EXITS(FUNC, EXIT)                                               \
  if (!EXPECT_EXITS(FUNC, EXIT))                                               \
  return

#define EXPECT_DEATH(FUNC, SIG)                                                \
  __llvm_libc::testing::Test::testProcessKilled(                               \
      Ctx, __llvm_libc::testing::Test::createCallable(FUNC), SIG, #FUNC, #SIG, \
      __FILE__, __LINE__)

#define ASSERT_DEATH(FUNC, EXIT)                                               \
  if (!EXPECT_DEATH(FUNC, EXIT))                                               \
  return

#define __CAT1(a, b) a##b
#define __CAT(a, b) __CAT1(a, b)
#define UNIQUE_VAR(prefix) __CAT(prefix, __LINE__)

#define EXPECT_THAT(MATCH, MATCHER)                                            \
  do {                                                                         \
    auto UNIQUE_VAR(__matcher) = (MATCHER);                                    \
    __llvm_libc::testing::Test::testMatch(                                     \
        Ctx, UNIQUE_VAR(__matcher).match((MATCH)), UNIQUE_VAR(__matcher),      \
        #MATCH, #MATCHER, __FILE__, __LINE__);                                 \
  } while (0)

#define ASSERT_THAT(MATCH, MATCHER)                                            \
  do {                                                                         \
    auto UNIQUE_VAR(__matcher) = (MATCHER);                                    \
    if (!__llvm_libc::testing::Test::testMatch(                                \
            Ctx, UNIQUE_VAR(__matcher).match((MATCH)), UNIQUE_VAR(__matcher),  \
            #MATCH, #MATCHER, __FILE__, __LINE__))                             \
      return;                                                                  \
  } while (0)

#endif // LLVM_LIBC_UTILS_UNITTEST_H
