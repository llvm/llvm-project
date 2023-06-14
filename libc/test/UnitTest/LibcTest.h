//===-- Base class for libc unittests ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H
#define LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H

// This is defined as a simple macro in test.h so that it exists for platforms
// that don't use our test infrastructure. It's defined as a proper function
// below.
#ifdef libc_make_test_file_path
#undef libc_make_test_file_path
#endif // libc_make_test_file_path

// This is defined as a macro here to avoid namespace issues.
#define libc_make_test_file_path(file_name)                                    \
  (__llvm_libc::testing::libc_make_test_file_path_func(file_name))

// This file can only include headers from src/__support/ or test/UnitTest. No
// other headers should be included.

#include "PlatformDefs.h"

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/c_string.h"
#include "test/UnitTest/ExecuteFunction.h"
#include "test/UnitTest/TestLogger.h"

namespace __llvm_libc {
namespace testing {

// Only the following conditions are supported. Notice that we do not have
// a TRUE or FALSE condition. That is because, C library functions do not
// return boolean values, but use integral return values to indicate true or
// false conditions. Hence, it is more appropriate to use the other comparison
// conditions for such cases.
enum class TestCond { EQ, NE, LT, LE, GT, GE };

struct MatcherBase {
  virtual ~MatcherBase() {}
  virtual void explainError() { tlog << "unknown error\n"; }
  // Override and return true to skip `explainError` step.
  virtual bool is_silent() const { return false; }
};

template <typename T> struct Matcher : public MatcherBase {
  bool match(const T &t);
};

namespace internal {

// A simple location object to allow consistent passing of __FILE__ and
// __LINE__.
struct Location {
  Location(const char *file, int line) : file(file), line(line) {}
  const char *file;
  int line;
};

// Supports writing a failing Location to tlog.
TestLogger &operator<<(TestLogger &logger, Location Loc);

#define LIBC_TEST_LOC_()                                                       \
  __llvm_libc::testing::internal::Location(__FILE__, __LINE__)

// Object to forward custom logging after the EXPECT / ASSERT macros.
struct Message {
  template <typename T> Message &operator<<(T value) {
    tlog << value;
    return *this;
  }
};

// A trivial object to catch the Message, this enables custom logging and
// returning from the test function, see LIBC_TEST_SCAFFOLDING_ below.
struct Failure {
  void operator=(Message msg) {}
};

struct RunContext {
  enum class RunResult : bool { Pass, Fail };

  RunResult status() const { return Status; }

  void markFail() { Status = RunResult::Fail; }

private:
  RunResult Status = RunResult::Pass;
};

template <typename ValType>
bool test(RunContext *Ctx, TestCond Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, Location Loc);

} // namespace internal

// NOTE: One should not create instances and call methods on them directly. One
// should use the macros TEST or TEST_F to write test cases.
class Test {
  Test *Next = nullptr;
  internal::RunContext *Ctx = nullptr;

  void setContext(internal::RunContext *C) { Ctx = C; }

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
            cpp::enable_if_t<cpp::is_integral_v<ValType>, int> = 0>
  bool test(TestCond Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, internal::Location Loc) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, Loc);
  }

  template <typename ValType,
            cpp::enable_if_t<cpp::is_enum_v<ValType>, int> = 0>
  bool test(TestCond Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, internal::Location Loc) {
    return internal::test(Ctx, Cond, (long long)LHS, (long long)RHS, LHSStr,
                          RHSStr, Loc);
  }

  template <typename ValType,
            cpp::enable_if_t<cpp::is_pointer_v<ValType>, ValType> = nullptr>
  bool test(TestCond Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, internal::Location Loc) {
    return internal::test(Ctx, Cond, (unsigned long long)LHS,
                          (unsigned long long)RHS, LHSStr, RHSStr, Loc);
  }

  template <
      typename ValType,
      cpp::enable_if_t<cpp::is_same_v<ValType, __llvm_libc::cpp::string_view>,
                       int> = 0>
  bool test(TestCond Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, internal::Location Loc) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, Loc);
  }

  template <typename ValType,
            cpp::enable_if_t<cpp::is_same_v<ValType, __llvm_libc::cpp::string>,
                             int> = 0>
  bool test(TestCond Cond, ValType LHS, ValType RHS, const char *LHSStr,
            const char *RHSStr, internal::Location Loc) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, Loc);
  }

  bool testStrEq(const char *LHS, const char *RHS, const char *LHSStr,
                 const char *RHSStr, internal::Location Loc);

  bool testStrNe(const char *LHS, const char *RHS, const char *LHSStr,
                 const char *RHSStr, internal::Location Loc);

  bool testMatch(bool MatchResult, MatcherBase &Matcher, const char *LHSStr,
                 const char *RHSStr, internal::Location Loc);

  template <typename MatcherT, typename ValType>
  bool matchAndExplain(MatcherT &&Matcher, ValType Value,
                       const char *MatcherStr, const char *ValueStr,
                       internal::Location Loc) {
    return testMatch(Matcher.match(Value), Matcher, ValueStr, MatcherStr, Loc);
  }

  bool testProcessExits(testutils::FunctionCaller *Func, int ExitCode,
                        const char *LHSStr, const char *RHSStr,
                        internal::Location Loc);

  bool testProcessKilled(testutils::FunctionCaller *Func, int Signal,
                         const char *LHSStr, const char *RHSStr,
                         internal::Location Loc);

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

extern int argc;
extern char **argv;
extern char **envp;

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

CString libc_make_test_file_path_func(const char *file_name);

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

// The GNU compiler emits a warning if nested "if" statements are followed by
// an "else" statement and braces are not used to explicitly disambiguate the
// "else" binding.  This leads to problems with code like:
//
//   if (gate)
//     ASSERT_*(condition) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#define LIBC_AMBIGUOUS_ELSE_BLOCKER_                                           \
  switch (0)                                                                   \
  case 0:                                                                      \
  default:

// If RET_OR_EMPTY is the 'return' keyword we perform an early return which
// corresponds to an assert. If it is empty the execution continues, this
// corresponds to an expect.
//
// The 'else' clause must not be enclosed into braces so that the << operator
// can be used to fill the Message.
//
// TEST is usually implemented as a function performing checking logic and
// returning a boolean. This expression is responsible for logging the
// diagnostic in case of failure.
#define LIBC_TEST_SCAFFOLDING_(TEST, RET_OR_EMPTY)                             \
  LIBC_AMBIGUOUS_ELSE_BLOCKER_                                                 \
  if (TEST)                                                                    \
    ;                                                                          \
  else                                                                         \
    RET_OR_EMPTY __llvm_libc::testing::internal::Failure() =                   \
        __llvm_libc::testing::internal::Message()

#define LIBC_TEST_BINOP_(COND, LHS, RHS, RET_OR_EMPTY)                         \
  LIBC_TEST_SCAFFOLDING_(test(__llvm_libc::testing::TestCond::COND, LHS, RHS,  \
                              #LHS, #RHS, LIBC_TEST_LOC_()),                   \
                         RET_OR_EMPTY)

////////////////////////////////////////////////////////////////////////////////
// Binary operations corresponding to the TestCond enum.

#define EXPECT_EQ(LHS, RHS) LIBC_TEST_BINOP_(EQ, LHS, RHS, )
#define ASSERT_EQ(LHS, RHS) LIBC_TEST_BINOP_(EQ, LHS, RHS, return)

#define EXPECT_NE(LHS, RHS) LIBC_TEST_BINOP_(NE, LHS, RHS, )
#define ASSERT_NE(LHS, RHS) LIBC_TEST_BINOP_(NE, LHS, RHS, return)

#define EXPECT_LT(LHS, RHS) LIBC_TEST_BINOP_(LT, LHS, RHS, )
#define ASSERT_LT(LHS, RHS) LIBC_TEST_BINOP_(LT, LHS, RHS, return)

#define EXPECT_LE(LHS, RHS) LIBC_TEST_BINOP_(LE, LHS, RHS, )
#define ASSERT_LE(LHS, RHS) LIBC_TEST_BINOP_(LE, LHS, RHS, return)

#define EXPECT_GT(LHS, RHS) LIBC_TEST_BINOP_(GT, LHS, RHS, )
#define ASSERT_GT(LHS, RHS) LIBC_TEST_BINOP_(GT, LHS, RHS, return)

#define EXPECT_GE(LHS, RHS) LIBC_TEST_BINOP_(GE, LHS, RHS, )
#define ASSERT_GE(LHS, RHS) LIBC_TEST_BINOP_(GE, LHS, RHS, return)

////////////////////////////////////////////////////////////////////////////////
// Boolean checks are handled as comparison to the true / false values.

#define EXPECT_TRUE(VAL) EXPECT_EQ(VAL, true)
#define ASSERT_TRUE(VAL) ASSERT_EQ(VAL, true)

#define EXPECT_FALSE(VAL) EXPECT_EQ(VAL, false)
#define ASSERT_FALSE(VAL) ASSERT_EQ(VAL, false)

////////////////////////////////////////////////////////////////////////////////
// String checks.

#define LIBC_TEST_STR_(TEST_FUNC, LHS, RHS, RET_OR_EMPTY)                      \
  LIBC_TEST_SCAFFOLDING_(TEST_FUNC(LHS, RHS, #LHS, #RHS, LIBC_TEST_LOC_()),    \
                         RET_OR_EMPTY)

#define EXPECT_STREQ(LHS, RHS) LIBC_TEST_STR_(testStrEq, LHS, RHS, )
#define ASSERT_STREQ(LHS, RHS) LIBC_TEST_STR_(testStrEq, LHS, RHS, return)

#define EXPECT_STRNE(LHS, RHS) LIBC_TEST_STR_(testStrNe, LHS, RHS, )
#define ASSERT_STRNE(LHS, RHS) LIBC_TEST_STR_(testStrNe, LHS, RHS, return)

////////////////////////////////////////////////////////////////////////////////
// Subprocess checks.

#ifdef ENABLE_SUBPROCESS_TESTS

#define LIBC_TEST_PROCESS_(TEST_FUNC, FUNC, VALUE, RET_OR_EMPTY)               \
  LIBC_TEST_SCAFFOLDING_(                                                      \
      TEST_FUNC(__llvm_libc::testing::Test::createCallable(FUNC), VALUE,       \
                #FUNC, #VALUE, LIBC_TEST_LOC_()),                              \
      RET_OR_EMPTY)

#define EXPECT_EXITS(FUNC, EXIT)                                               \
  LIBC_TEST_PROCESS_(testProcessExits, FUNC, EXIT, )
#define ASSERT_EXITS(FUNC, EXIT)                                               \
  LIBC_TEST_PROCESS_(testProcessExits, FUNC, EXIT, return)

#define EXPECT_DEATH(FUNC, SIG)                                                \
  LIBC_TEST_PROCESS_(testProcessKilled, FUNC, SIG, )
#define ASSERT_DEATH(FUNC, SIG)                                                \
  LIBC_TEST_PROCESS_(testProcessKilled, FUNC, SIG, return)

#endif // ENABLE_SUBPROCESS_TESTS

////////////////////////////////////////////////////////////////////////////////
// Custom matcher checks.

#define LIBC_TEST_MATCH_(MATCHER, MATCH, MATCHER_STR, MATCH_STR, RET_OR_EMPTY) \
  LIBC_TEST_SCAFFOLDING_(matchAndExplain(MATCHER, MATCH, MATCHER_STR,          \
                                         MATCH_STR, LIBC_TEST_LOC_()),         \
                         RET_OR_EMPTY)

#define EXPECT_THAT(MATCH, MATCHER)                                            \
  LIBC_TEST_MATCH_(MATCHER, MATCH, #MATCHER, #MATCH, )
#define ASSERT_THAT(MATCH, MATCHER)                                            \
  LIBC_TEST_MATCH_(MATCHER, MATCH, #MATCHER, #MATCH, return)

#define WITH_SIGNAL(X) X

#endif // LLVM_LIBC_UTILS_UNITTEST_LIBCTEST_H
