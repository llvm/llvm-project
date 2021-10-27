//===-- Base class for libc unittests ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H
#define LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H

// This file can only include headers from src/__support/CPP/ or
// utils/testutils. No other headers should be included.

#include "PlatformDefs.h"

#include "src/__support/CPP/TypeTraits.h"
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
bool test(RunContext *Ctx, TestCondition Cond, ValType LHS, ValType RHS,
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
  RunContext *Ctx = nullptr;

  void setContext(RunContext *C) { Ctx = C; }

public:
  virtual ~Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runTests(const char *);

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
  bool test(TestCondition Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, File, Line);
  }

  template <
      typename ValType,
      cpp::EnableIfType<cpp::IsPointerType<ValType>::Value, ValType> = nullptr>
  bool test(TestCondition Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, (unsigned long long)LHS,
                          (unsigned long long)RHS, LHSStr, RHSStr, File, Line);
  }

  bool testStrEq(const char *LHS, const char *RHS, const char *LHSStr,
                 const char *RHSStr, const char *File, unsigned long Line);

  bool testStrNe(const char *LHS, const char *RHS, const char *LHSStr,
                 const char *RHSStr, const char *File, unsigned long Line);

  bool testMatch(bool MatchResult, MatcherBase &Matcher, const char *LHSStr,
                 const char *RHSStr, const char *File, unsigned long Line);

  bool testProcessExits(testutils::FunctionCaller *Func, int ExitCode,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

  bool testProcessKilled(testutils::FunctionCaller *Func, int Signal,
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
  virtual void Run() = 0;
  virtual const char *getName() const = 0;

  static Test *Start;
  static Test *End;
};

namespace internal {

constexpr bool same_prefix(char const *lhs, char const *rhs, int const len) {
  for (int i = 0; (*lhs || *rhs) && (i < len); ++lhs, ++rhs, ++i)
    if (*lhs != *rhs)
      return false;
  return true;
}

constexpr bool valid_prefix(char const *lhs) {
  return same_prefix(lhs, "LlvmLibc", 8);
}

// 'str' is a null terminated string of the form
// "const char *__llvm_libc::testing::internal::GetTypeName() [ParamType = XXX]"
// We return the substring that start at character '[' or a default message.
constexpr char const *GetPrettyFunctionParamType(char const *str) {
  for (const char *ptr = str; *ptr != '\0'; ++ptr)
    if (*ptr == '[')
      return ptr;
  return "UNSET : declare with REGISTER_TYPE_NAME";
}

// This function recovers ParamType at compile time by using __PRETTY_FUNCTION__
// It can be customized by using the REGISTER_TYPE_NAME macro below.
template <typename ParamType> static constexpr const char *GetTypeName() {
  return GetPrettyFunctionParamType(__PRETTY_FUNCTION__);
}

template <typename T>
static inline void GenerateName(char *buffer, int buffer_size,
                                const char *prefix) {
  if (buffer_size == 0)
    return;

  // Make sure string is null terminated.
  --buffer_size;
  buffer[buffer_size] = '\0';

  const auto AppendChar = [&](char c) {
    if (buffer_size > 0) {
      *buffer = c;
      ++buffer;
      --buffer_size;
    }
  };
  const auto AppendStr = [&](const char *str) {
    for (; str && *str != '\0'; ++str)
      AppendChar(*str);
  };

  AppendStr(prefix);
  AppendChar(' ');
  AppendStr(GetTypeName<T>());
  AppendChar('\0');
}

// TestCreator implements a linear hierarchy of test instances, effectively
// instanciating all tests with Types in a single object.
template <template <typename> class TemplatedTestClass, typename... Types>
struct TestCreator;

template <template <typename> class TemplatedTestClass, typename Head,
          typename... Tail>
struct TestCreator<TemplatedTestClass, Head, Tail...>
    : private TestCreator<TemplatedTestClass, Tail...> {
  TemplatedTestClass<Head> instance;
};

template <template <typename> class TemplatedTestClass>
struct TestCreator<TemplatedTestClass> {};

// A type list to declare the set of types to instantiate the tests with.
template <typename... Types> struct TypeList {
  template <template <typename> class TemplatedTestClass> struct Tests {
    using type = TestCreator<TemplatedTestClass, Types...>;
  };
};

} // namespace internal

// Make TypeList visible in __llvm_libc::testing.
template <typename... Types> using TypeList = internal::TypeList<Types...>;

} // namespace testing
} // namespace __llvm_libc

// For TYPED_TEST and TYPED_TEST_F below we need to display which type was used
// to run the test. The default will return the fully qualified canonical type
// but it can be difficult to read. We provide the following macro to allow the
// client to register the type name as they see it in the code.
#define REGISTER_TYPE_NAME(TYPE)                                               \
  template <>                                                                  \
  constexpr const char *__llvm_libc::testing::internal::GetTypeName<TYPE>() {  \
    return "[ParamType = " #TYPE "]";                                          \
  }

#define TYPED_TEST(SuiteName, TestName, TypeList)                              \
  static_assert(                                                               \
      __llvm_libc::testing::internal::valid_prefix(#SuiteName),                \
      "All LLVM-libc TYPED_TEST suite names must start with 'LlvmLibc'.");     \
  template <typename T>                                                        \
  class SuiteName##_##TestName : public __llvm_libc::testing::Test {           \
  public:                                                                      \
    using ParamType = T;                                                       \
    char name[256];                                                            \
    SuiteName##_##TestName() {                                                 \
      addTest(this);                                                           \
      __llvm_libc::testing::internal::GenerateName<T>(                         \
          name, sizeof(name), #SuiteName "." #TestName);                       \
    }                                                                          \
    void Run() override;                                                       \
    const char *getName() const override { return name; }                      \
  };                                                                           \
  TypeList::Tests<SuiteName##_##TestName>::type                                \
      SuiteName##_##TestName##_Instance;                                       \
  template <typename T> void SuiteName##_##TestName<T>::Run()

#define TYPED_TEST_F(SuiteClass, TestName, TypeList)                           \
  static_assert(__llvm_libc::testing::internal::valid_prefix(#SuiteClass),     \
                "All LLVM-libc TYPED_TEST_F suite class names must start "     \
                "with 'LlvmLibc'.");                                           \
  template <typename T> class SuiteClass##_##TestName : public SuiteClass<T> { \
  public:                                                                      \
    using ParamType = T;                                                       \
    char name[256];                                                            \
    SuiteClass##_##TestName() {                                                \
      SuiteClass<T>::addTest(this);                                            \
      __llvm_libc::testing::internal::GenerateName<T>(                         \
          name, sizeof(name), #SuiteClass "." #TestName);                      \
    }                                                                          \
    void Run() override;                                                       \
    const char *getName() const override { return name; }                      \
  };                                                                           \
  TypeList::Tests<SuiteClass##_##TestName>::type                               \
      SuiteClass##_##TestName##_Instance;                                      \
  template <typename T> void SuiteClass##_##TestName<T>::Run()

#define TEST(SuiteName, TestName)                                              \
  static_assert(__llvm_libc::testing::internal::valid_prefix(#SuiteName),      \
                "All LLVM-libc TEST suite names must start with 'LlvmLibc'."); \
  class SuiteName##_##TestName : public __llvm_libc::testing::Test {           \
  public:                                                                      \
    SuiteName##_##TestName() { addTest(this); }                                \
    void Run() override;                                                       \
    const char *getName() const override { return #SuiteName "." #TestName; }  \
  };                                                                           \
  SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
  void SuiteName##_##TestName::Run()

#define TEST_F(SuiteClass, TestName)                                           \
  static_assert(                                                               \
      __llvm_libc::testing::internal::valid_prefix(#SuiteClass),               \
      "All LLVM-libc TEST_F suite class names must start with 'LlvmLibc'.");   \
  class SuiteClass##_##TestName : public SuiteClass {                          \
  public:                                                                      \
    SuiteClass##_##TestName() { addTest(this); }                               \
    void Run() override;                                                       \
    const char *getName() const override { return #SuiteClass "." #TestName; } \
  };                                                                           \
  SuiteClass##_##TestName SuiteClass##_##TestName##_Instance;                  \
  void SuiteClass##_##TestName::Run()

#define EXPECT_EQ(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_EQ, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_EQ(LHS, RHS)                                                    \
  if (!EXPECT_EQ(LHS, RHS))                                                    \
  return

#define EXPECT_NE(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_NE, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_NE(LHS, RHS)                                                    \
  if (!EXPECT_NE(LHS, RHS))                                                    \
  return

#define EXPECT_LT(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_LT, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_LT(LHS, RHS)                                                    \
  if (!EXPECT_LT(LHS, RHS))                                                    \
  return

#define EXPECT_LE(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_LE, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_LE(LHS, RHS)                                                    \
  if (!EXPECT_LE(LHS, RHS))                                                    \
  return

#define EXPECT_GT(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_GT, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_GT(LHS, RHS)                                                    \
  if (!EXPECT_GT(LHS, RHS))                                                    \
  return

#define EXPECT_GE(LHS, RHS)                                                    \
  this->test(__llvm_libc::testing::Cond_GE, (LHS), (RHS), #LHS, #RHS,          \
             __FILE__, __LINE__)
#define ASSERT_GE(LHS, RHS)                                                    \
  if (!EXPECT_GE(LHS, RHS))                                                    \
  return

#define EXPECT_STREQ(LHS, RHS)                                                 \
  this->testStrEq((LHS), (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_STREQ(LHS, RHS)                                                 \
  if (!EXPECT_STREQ(LHS, RHS))                                                 \
  return

#define EXPECT_STRNE(LHS, RHS)                                                 \
  this->testStrNe((LHS), (RHS), #LHS, #RHS, __FILE__, __LINE__)
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

#ifdef ENABLE_SUBPROCESS_TESTS

#define EXPECT_EXITS(FUNC, EXIT)                                               \
  this->testProcessExits(__llvm_libc::testing::Test::createCallable(FUNC),     \
                         EXIT, #FUNC, #EXIT, __FILE__, __LINE__)

#define ASSERT_EXITS(FUNC, EXIT)                                               \
  if (!EXPECT_EXITS(FUNC, EXIT))                                               \
  return

#define EXPECT_DEATH(FUNC, SIG)                                                \
  this->testProcessKilled(__llvm_libc::testing::Test::createCallable(FUNC),    \
                          SIG, #FUNC, #SIG, __FILE__, __LINE__)

#define ASSERT_DEATH(FUNC, EXIT)                                               \
  if (!EXPECT_DEATH(FUNC, EXIT))                                               \
  return

#endif // ENABLE_SUBPROCESS_TESTS

#define __CAT1(a, b) a##b
#define __CAT(a, b) __CAT1(a, b)
#define UNIQUE_VAR(prefix) __CAT(prefix, __LINE__)

#define EXPECT_THAT(MATCH, MATCHER)                                            \
  do {                                                                         \
    auto UNIQUE_VAR(__matcher) = (MATCHER);                                    \
    this->testMatch(UNIQUE_VAR(__matcher).match((MATCH)),                      \
                    UNIQUE_VAR(__matcher), #MATCH, #MATCHER, __FILE__,         \
                    __LINE__);                                                 \
  } while (0)

#define ASSERT_THAT(MATCH, MATCHER)                                            \
  do {                                                                         \
    auto UNIQUE_VAR(__matcher) = (MATCHER);                                    \
    if (!this->testMatch(UNIQUE_VAR(__matcher).match((MATCH)),                 \
                         UNIQUE_VAR(__matcher), #MATCH, #MATCHER, __FILE__,    \
                         __LINE__))                                            \
      return;                                                                  \
  } while (0)

#define WITH_SIGNAL(X) X

#endif // LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H
