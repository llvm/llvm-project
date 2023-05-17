// RUN: %check_clang_tidy %s google-upgrade-googletest-case %t -- -- -I%S/Inputs
// RUN: %check_clang_tidy -check-suffix=NOSUITE %s google-upgrade-googletest-case %t -- -- -DNOSUITE -I%S/Inputs/gtest/nosuite

#include "gtest/gtest.h"

// When including a version of googletest without the replacement names, this
// check should not produce any diagnostics. The following dummy fix is present
// because `check_clang_tidy.py` requires at least one warning, fix or note.
void Dummy() {}
// CHECK-FIXES-NOSUITE: void Dummy() {}

// ----------------------------------------------------------------------------
// Macros

TYPED_TEST_CASE(FooTest, FooTypes);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: Google Test APIs named with 'case' are deprecated; use equivalent APIs named with 'suite'
// CHECK-FIXES: TYPED_TEST_SUITE(FooTest, FooTypes);
TYPED_TEST_CASE_P(FooTest);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: Google Test APIs named with 'case'
// CHECK-FIXES: TYPED_TEST_SUITE_P(FooTest);
REGISTER_TYPED_TEST_CASE_P(FooTest, FooTestName);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: Google Test APIs named with 'case'
// CHECK-FIXES: REGISTER_TYPED_TEST_SUITE_P(FooTest, FooTestName);
INSTANTIATE_TYPED_TEST_CASE_P(FooPrefix, FooTest, FooTypes);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: Google Test APIs named with 'case'
// CHECK-FIXES: INSTANTIATE_TYPED_TEST_SUITE_P(FooPrefix, FooTest, FooTypes);

#ifdef TYPED_TEST_CASE
// CHECK-MESSAGES: [[@LINE-1]]:2: warning: Google Test APIs named with 'case'
#undef TYPED_TEST_CASE
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
#define TYPED_TEST_CASE(CaseName, Types, ...)
#endif

#ifdef TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:2: warning: Google Test APIs named with 'case'
#undef TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
#define TYPED_TEST_CASE_P(SuiteName)
#endif

#ifdef REGISTER_TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:2: warning: Google Test APIs named with 'case'
#undef REGISTER_TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
#define REGISTER_TYPED_TEST_CASE_P(SuiteName, ...)
#endif

#ifdef INSTANTIATE_TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:2: warning: Google Test APIs named with 'case'
#undef INSTANTIATE_TYPED_TEST_CASE_P
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
#define INSTANTIATE_TYPED_TEST_CASE_P(Prefix, SuiteName, Types, ...)
#endif

TYPED_TEST_CASE(FooTest, FooTypes);
TYPED_TEST_CASE_P(FooTest);
REGISTER_TYPED_TEST_CASE_P(FooTest, FooTestName);
INSTANTIATE_TYPED_TEST_CASE_P(FooPrefix, FooTest, FooTypes);

// ----------------------------------------------------------------------------
// testing::Test

class FooTest : public testing::Test {
public:
  static void SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: static void SetUpTestSuite();
  static void TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: static void TearDownTestSuite();
};

void FooTest::SetUpTestCase() {}
// CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTest::SetUpTestSuite() {}

void FooTest::TearDownTestCase() {}
// CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTest::TearDownTestSuite() {}

template <typename T> class FooTypedTest : public testing::Test {
public:
  static void SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: static void SetUpTestSuite();
  static void TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: static void TearDownTestSuite();
};

template <typename T> void FooTypedTest<T>::SetUpTestCase() {}
// CHECK-MESSAGES: [[@LINE-1]]:45: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTypedTest<T>::SetUpTestSuite() {}

template <typename T> void FooTypedTest<T>::TearDownTestCase() {}
// CHECK-MESSAGES: [[@LINE-1]]:45: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTypedTest<T>::TearDownTestSuite() {}

class BarTest : public testing::Test {
public:
  using Test::SetUpTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using Test::SetUpTestSuite;
  using Test::TearDownTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using Test::TearDownTestSuite;
};

class BarTest2 : public FooTest {
public:
  using FooTest::SetUpTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using FooTest::SetUpTestSuite;
  using FooTest::TearDownTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using FooTest::TearDownTestSuite;
};

// If a derived type already has the replacements, we only provide a warning
// since renaming or deleting the old declarations may not be safe.
class BarTest3 : public testing::Test {
 public:
  static void SetUpTestCase() {}
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  static void SetUpTestSuite() {}

  static void TearDownTestCase() {}
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  static void TearDownTestSuite() {}
};

namespace nesting_ns {
namespace testing {

class Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

} // namespace testing

void Test() {
  testing::Test::SetUpTestCase();
  testing::Test::TearDownTestCase();
}

} // namespace nesting_ns

template <typename T>
void testInstantiationOnlyWarns() {
  T::SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: Google Test APIs named with 'case'
  T::TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: Google Test APIs named with 'case'
}

#define SET_UP_TEST_CASE_MACRO_REPLACE SetUpTestCase
#define TEST_SET_UP_TEST_CASE_MACRO_WARN_ONLY ::testing::Test::SetUpTestCase

void setUpTearDownCallAndReference() {
  testing::Test::SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::Test::SetUpTestSuite();
  FooTest::SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: FooTest::SetUpTestSuite();

  testing::Test::TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::Test::TearDownTestSuite();
  FooTest::TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: FooTest::TearDownTestSuite();

  auto F = &testing::Test::SetUpTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F = &testing::Test::SetUpTestSuite;
  F = &testing::Test::TearDownTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: F = &testing::Test::TearDownTestSuite;
  F = &FooTest::SetUpTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: F = &FooTest::SetUpTestSuite;
  F = &FooTest::TearDownTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: F = &FooTest::TearDownTestSuite;

  using MyTest = testing::Test;
  MyTest::SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: MyTest::SetUpTestSuite();
  MyTest::TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: MyTest::TearDownTestSuite();

  BarTest3::SetUpTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: BarTest3::SetUpTestSuite();
  BarTest3::TearDownTestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: BarTest3::TearDownTestSuite();

  testInstantiationOnlyWarns<testing::Test>();

  testing::Test::SET_UP_TEST_CASE_MACRO_REPLACE();
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::Test::SetUpTestSuite();
  TEST_SET_UP_TEST_CASE_MACRO_WARN_ONLY();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
}

// ----------------------------------------------------------------------------
// testing::TestInfo

class FooTestInfo : public testing::TestInfo {
public:
  const char *test_case_name() const;
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const char *test_suite_name() const;
};

const char *FooTestInfo::test_case_name() const {}
// CHECK-MESSAGES: [[@LINE-1]]:26: warning: Google Test APIs named with 'case'
// CHECK-FIXES: const char *FooTestInfo::test_suite_name() const {}

class BarTestInfo : public testing::TestInfo {
public:
  using TestInfo::test_case_name;
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using TestInfo::test_suite_name;
};

class BarTestInfo2 : public FooTestInfo {
public:
  using FooTestInfo::test_case_name;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using FooTestInfo::test_suite_name;
};

class BarTestInfo3 : public testing::TestInfo {
 public:
  const char* test_case_name() const;
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: Google Test APIs named with 'case'
  const char* test_suite_name() const;
};

namespace nesting_ns {
namespace testing {

class TestInfo {
public:
  const char *test_case_name() const;
};

} // namespace testing

void FuncInfo() {
  testing::TestInfo t;
  (void)t.test_case_name();
}

} // namespace nesting_ns

template <typename T>
void testInfoInstantiationOnlyWarns() {
  T t;
  (void)t.test_case_name();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
}

#define TEST_CASE_NAME_MACRO_REPLACE test_case_name
#define TEST_CASE_NAME_MACRO_WARN_ONLY testing::TestInfo().test_case_name

void testInfoCallAndReference() {
  (void)testing::TestInfo().test_case_name();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::TestInfo().test_suite_name();
  (void)FooTestInfo().test_case_name();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooTestInfo().test_suite_name();
  auto F1 = &testing::TestInfo::test_case_name;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F1 = &testing::TestInfo::test_suite_name;
  auto F2 = &FooTestInfo::test_case_name;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F2 = &FooTestInfo::test_suite_name;
  using MyTestInfo = testing::TestInfo;
  (void)MyTestInfo().test_case_name();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyTestInfo().test_suite_name();
  (void)BarTestInfo3().test_case_name();
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)BarTestInfo3().test_suite_name();

  testInfoInstantiationOnlyWarns<testing::TestInfo>();

  (void)testing::TestInfo().TEST_CASE_NAME_MACRO_REPLACE();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::TestInfo().test_suite_name();
  (void)TEST_CASE_NAME_MACRO_WARN_ONLY();
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: Google Test APIs named with 'case'
}

// ----------------------------------------------------------------------------
// testing::TestEventListener

class FooTestEventListener : public testing::TestEventListener {
public:
  void OnTestCaseStart(const testing::TestCase &) override;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:39: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: void OnTestSuiteStart(const testing::TestSuite &) override;
  void OnTestCaseEnd(const testing::TestCase &) override;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:37: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: void OnTestSuiteEnd(const testing::TestSuite &) override;
};

void FooTestEventListener::OnTestCaseStart(const testing::TestCase &) {}
// CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
// CHECK-MESSAGES: [[@LINE-2]]:59: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTestEventListener::OnTestSuiteStart(const testing::TestSuite &) {}

void FooTestEventListener::OnTestCaseEnd(const testing::TestCase &) {}
// CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
// CHECK-MESSAGES: [[@LINE-2]]:57: warning: Google Test APIs named with 'case'
// CHECK-FIXES: void FooTestEventListener::OnTestSuiteEnd(const testing::TestSuite &) {}

class BarTestEventListener : public testing::TestEventListener {
public:
  using TestEventListener::OnTestCaseStart;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using TestEventListener::OnTestSuiteStart;
  using TestEventListener::OnTestCaseEnd;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using TestEventListener::OnTestSuiteEnd;
};

class BarTestEventListener2 : public BarTestEventListener {
public:
  using BarTestEventListener::OnTestCaseStart;
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarTestEventListener::OnTestSuiteStart;
  using BarTestEventListener::OnTestCaseEnd;
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarTestEventListener::OnTestSuiteEnd;
};

#ifndef NOSUITE

class BarTestEventListener3 : public testing::TestEventListener {
public:
  void OnTestCaseStart(const testing::TestSuite &) override;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
  void OnTestSuiteStart(const testing::TestSuite &) override;

  void OnTestCaseEnd(const testing::TestSuite &) override;
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: Google Test APIs named with 'case'
  void OnTestSuiteEnd(const testing::TestSuite &) override;
};

#endif

namespace nesting_ns {
namespace testing {

class TestEventListener {
public:
  virtual void OnTestCaseStart(const ::testing::TestCase &);
  // CHECK-MESSAGES: [[@LINE-1]]:49: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: virtual void OnTestCaseStart(const ::testing::TestSuite &);
  virtual void OnTestCaseEnd(const ::testing::TestCase &);
  // CHECK-MESSAGES: [[@LINE-1]]:47: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: virtual void OnTestCaseEnd(const ::testing::TestSuite &);
};

} // namespace testing

void FuncTestEventListener(::testing::TestCase &Case) {
  // CHECK-MESSAGES: [[@LINE-1]]:39: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: void FuncTestEventListener(::testing::TestSuite &Case) {
  testing::TestEventListener().OnTestCaseStart(Case);
  testing::TestEventListener().OnTestCaseEnd(Case);
}

} // namespace nesting_ns

#ifndef NOSUITE

template <typename T>
void testEventListenerInstantiationOnlyWarns() {
  T().OnTestCaseStart(testing::TestSuite());
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  T().OnTestCaseEnd(testing::TestSuite());
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
}

#endif

#define ON_TEST_CASE_START_MACRO_REPLACE OnTestCaseStart
#define ON_TEST_CASE_START_MACRO_WARN_ONLY                                     \
  testing::TestEventListener().OnTestCaseStart

#define ON_TEST_CASE_END_MACRO_REPLACE OnTestCaseEnd
#define ON_TEST_CASE_END_MACRO_WARN_ONLY                                       \
  testing::TestEventListener().OnTestCaseEnd

void testEventListenerCallAndReference(testing::TestCase &Case) {
  // CHECK-MESSAGES: [[@LINE-1]]:49: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: void testEventListenerCallAndReference(testing::TestSuite &Case) {
  testing::TestEventListener().OnTestCaseStart(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestEventListener().OnTestSuiteStart(Case);
  testing::TestEventListener().OnTestCaseEnd(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestEventListener().OnTestSuiteEnd(Case);

  FooTestEventListener().OnTestCaseStart(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: FooTestEventListener().OnTestSuiteStart(Case);
  FooTestEventListener().OnTestCaseEnd(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: FooTestEventListener().OnTestSuiteEnd(Case);

  auto F1 = &testing::TestEventListener::OnTestCaseStart;
  // CHECK-MESSAGES: [[@LINE-1]]:42: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F1 = &testing::TestEventListener::OnTestSuiteStart;
  F1 = &testing::TestEventListener::OnTestCaseEnd;
  // CHECK-MESSAGES: [[@LINE-1]]:37: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: F1 = &testing::TestEventListener::OnTestSuiteEnd;

  auto F2 = &FooTestEventListener::OnTestCaseStart;
  // CHECK-MESSAGES: [[@LINE-1]]:36: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F2 = &FooTestEventListener::OnTestSuiteStart;
  F2 = &FooTestEventListener::OnTestCaseEnd;
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: F2 = &FooTestEventListener::OnTestSuiteEnd;

#ifndef NOSUITE

  BarTestEventListener3().OnTestCaseStart(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: BarTestEventListener3().OnTestSuiteStart(Case);
  BarTestEventListener3().OnTestCaseEnd(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: BarTestEventListener3().OnTestSuiteEnd(Case);

  testEventListenerInstantiationOnlyWarns<testing::TestEventListener>();

#endif

  testing::TestEventListener().ON_TEST_CASE_START_MACRO_REPLACE(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestEventListener().OnTestSuiteStart(Case);
  ON_TEST_CASE_START_MACRO_WARN_ONLY(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'

  testing::TestEventListener().ON_TEST_CASE_END_MACRO_REPLACE(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestEventListener().OnTestSuiteEnd(Case);
  ON_TEST_CASE_END_MACRO_WARN_ONLY(Case);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
}

// ----------------------------------------------------------------------------
// testing::UnitTest

class FooUnitTest : public testing::UnitTest {
public:
  testing::TestCase *current_test_case() const;
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestSuite *current_test_suite() const;
  int successful_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: int successful_test_suite_count() const;
  int failed_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: int failed_test_suite_count() const;
  int total_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: int total_test_suite_count() const;
  int test_case_to_run_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: int test_suite_to_run_count() const;
  const testing::TestCase *GetTestCase(int) const;
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const testing::TestSuite *GetTestSuite(int) const;
};

testing::TestCase *FooUnitTest::current_test_case() const {}
// CHECK-MESSAGES: [[@LINE-1]]:10: warning: Google Test APIs named with 'case'
// CHECK-MESSAGES: [[@LINE-2]]:33: warning: Google Test APIs named with 'case'
// CHECK-FIXES: testing::TestSuite *FooUnitTest::current_test_suite() const {}
int FooUnitTest::successful_test_case_count() const {}
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
// CHECK-FIXES: int FooUnitTest::successful_test_suite_count() const {}
int FooUnitTest::failed_test_case_count() const {}
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
// CHECK-FIXES: int FooUnitTest::failed_test_suite_count() const {}
int FooUnitTest::total_test_case_count() const {}
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
// CHECK-FIXES: int FooUnitTest::total_test_suite_count() const {}
int FooUnitTest::test_case_to_run_count() const {}
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
// CHECK-FIXES: int FooUnitTest::test_suite_to_run_count() const {}
const testing::TestCase *FooUnitTest::GetTestCase(int) const {}
// CHECK-MESSAGES: [[@LINE-1]]:16: warning: Google Test APIs named with 'case'
// CHECK-MESSAGES: [[@LINE-2]]:39: warning: Google Test APIs named with 'case'
// CHECK-FIXES: const testing::TestSuite *FooUnitTest::GetTestSuite(int) const {}

// Type derived from testing::TestCase
class BarUnitTest : public testing::UnitTest {
public:
  using testing::UnitTest::current_test_case;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::current_test_suite;
  using testing::UnitTest::successful_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::successful_test_suite_count;
  using testing::UnitTest::failed_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::failed_test_suite_count;
  using testing::UnitTest::total_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::total_test_suite_count;
  using testing::UnitTest::test_case_to_run_count;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::test_suite_to_run_count;
  using testing::UnitTest::GetTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using testing::UnitTest::GetTestSuite;
};

class BarUnitTest2 : public BarUnitTest {
  using BarUnitTest::current_test_case;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::current_test_suite;
  using BarUnitTest::successful_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::successful_test_suite_count;
  using BarUnitTest::failed_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::failed_test_suite_count;
  using BarUnitTest::total_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::total_test_suite_count;
  using BarUnitTest::test_case_to_run_count;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::test_suite_to_run_count;
  using BarUnitTest::GetTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: using BarUnitTest::GetTestSuite;
};

#ifndef NOSUITE

class BarUnitTest3 : public testing::UnitTest {
  testing::TestSuite *current_test_case() const;
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  int successful_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  int failed_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  int total_test_case_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  int test_case_to_run_count() const;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  const testing::TestSuite *GetTestCase(int) const;
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'

  testing::TestSuite *current_test_suite() const;
  int successful_test_suite_count() const;
  int failed_test_suite_count() const;
  int total_test_suite_count() const;
  int test_suite_to_run_count() const;
  const testing::TestSuite *GetTestSuite(int) const;
};

#endif

namespace nesting_ns {
namespace testing {

class TestSuite;

class UnitTest {
public:
  TestSuite *current_test_case() const;
  int successful_test_case_count() const;
  int failed_test_case_count() const;
  int total_test_case_count() const;
  int test_case_to_run_count() const;
  const TestSuite *GetTestCase(int) const;
};

} // namespace testing

void FuncUnitTest() {
  testing::UnitTest t;
  (void)t.current_test_case();
  (void)t.successful_test_case_count();
  (void)t.failed_test_case_count();
  (void)t.total_test_case_count();
  (void)t.test_case_to_run_count();
  (void)t.GetTestCase(0);
}

} // namespace nesting_ns

template <typename T>
void unitTestInstantiationOnlyWarns() {
  T t;
  (void)t.current_test_case();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  (void)t.successful_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  (void)t.failed_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  (void)t.total_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  (void)t.test_case_to_run_count();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
  (void)t.GetTestCase(0);
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
}

#define UNIT_TEST_NAME_MACRO_REPLACE1 current_test_case
#define UNIT_TEST_NAME_MACRO_REPLACE2 successful_test_case_count
#define UNIT_TEST_NAME_MACRO_REPLACE3 failed_test_case_count
#define UNIT_TEST_NAME_MACRO_REPLACE4 total_test_case_count
#define UNIT_TEST_NAME_MACRO_REPLACE5 test_case_to_run_count
#define UNIT_TEST_NAME_MACRO_REPLACE6 GetTestCase
#define UNIT_TEST_NAME_MACRO_WARN_ONLY1 testing::UnitTest().current_test_case
#define UNIT_TEST_NAME_MACRO_WARN_ONLY2                                        \
  testing::UnitTest().successful_test_case_count
#define UNIT_TEST_NAME_MACRO_WARN_ONLY3                                        \
  testing::UnitTest().failed_test_case_count
#define UNIT_TEST_NAME_MACRO_WARN_ONLY4                                        \
  testing::UnitTest().total_test_case_count
#define UNIT_TEST_NAME_MACRO_WARN_ONLY5                                        \
  testing::UnitTest().test_case_to_run_count
#define UNIT_TEST_NAME_MACRO_WARN_ONLY6 testing::UnitTest().GetTestCase

void unitTestCallAndReference() {
  (void)testing::UnitTest().current_test_case();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().current_test_suite();
  (void)testing::UnitTest().successful_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().successful_test_suite_count();
  (void)testing::UnitTest().failed_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().failed_test_suite_count();
  (void)testing::UnitTest().total_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().total_test_suite_count();
  (void)testing::UnitTest().test_case_to_run_count();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().test_suite_to_run_count();
  (void)testing::UnitTest().GetTestCase(0);
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().GetTestSuite(0);

  (void)FooUnitTest().current_test_case();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().current_test_suite();
  (void)FooUnitTest().successful_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().successful_test_suite_count();
  (void)FooUnitTest().failed_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().failed_test_suite_count();
  (void)FooUnitTest().total_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().total_test_suite_count();
  (void)FooUnitTest().test_case_to_run_count();
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().test_suite_to_run_count();
  (void)FooUnitTest().GetTestCase(0);
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)FooUnitTest().GetTestSuite(0);

  auto U1 = &testing::UnitTest::current_test_case;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U1 = &testing::UnitTest::current_test_suite;
  auto U2 = &testing::UnitTest::successful_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U2 = &testing::UnitTest::successful_test_suite_count;
  auto U3 = &testing::UnitTest::failed_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U3 = &testing::UnitTest::failed_test_suite_count;
  auto U4 = &testing::UnitTest::total_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U4 = &testing::UnitTest::total_test_suite_count;
  auto U5 = &testing::UnitTest::test_case_to_run_count;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U5 = &testing::UnitTest::test_suite_to_run_count;
  auto U6 = &testing::UnitTest::GetTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto U6 = &testing::UnitTest::GetTestSuite;

  auto F1 = &FooUnitTest::current_test_case;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F1 = &FooUnitTest::current_test_suite;
  auto F2 = &FooUnitTest::successful_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F2 = &FooUnitTest::successful_test_suite_count;
  auto F3 = &FooUnitTest::failed_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F3 = &FooUnitTest::failed_test_suite_count;
  auto F4 = &FooUnitTest::total_test_case_count;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F4 = &FooUnitTest::total_test_suite_count;
  auto F5 = &FooUnitTest::test_case_to_run_count;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F5 = &FooUnitTest::test_suite_to_run_count;
  auto F6 = &FooUnitTest::GetTestCase;
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: auto F6 = &FooUnitTest::GetTestSuite;

  using MyUnitTest = testing::UnitTest;
  (void)MyUnitTest().current_test_case();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().current_test_suite();
  (void)MyUnitTest().successful_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().successful_test_suite_count();
  (void)MyUnitTest().failed_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().failed_test_suite_count();
  (void)MyUnitTest().total_test_case_count();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().total_test_suite_count();
  (void)MyUnitTest().test_case_to_run_count();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().test_suite_to_run_count();
  (void)MyUnitTest().GetTestCase(0);
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)MyUnitTest().GetTestSuite(0);

  unitTestInstantiationOnlyWarns<testing::UnitTest>();

  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE1();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().current_test_suite();
  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE2();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().successful_test_suite_count();
  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE3();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().failed_test_suite_count();
  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE4();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().total_test_suite_count();
  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE5();
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().test_suite_to_run_count();
  (void)testing::UnitTest().UNIT_TEST_NAME_MACRO_REPLACE6(0);
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)testing::UnitTest().GetTestSuite(0);

  UNIT_TEST_NAME_MACRO_WARN_ONLY1();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  UNIT_TEST_NAME_MACRO_WARN_ONLY2();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  UNIT_TEST_NAME_MACRO_WARN_ONLY3();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  UNIT_TEST_NAME_MACRO_WARN_ONLY4();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  UNIT_TEST_NAME_MACRO_WARN_ONLY5();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  UNIT_TEST_NAME_MACRO_WARN_ONLY6(0);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
}

// ----------------------------------------------------------------------------
// testing::TestCase

template <typename T>
void TestCaseInTemplate() {
  T t;

  testing::TestCase Case;
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestSuite Case;
}

#define TEST_CASE_CAN_FIX TestCase
#define TEST_CASE_WARN_ONLY testing::TestCase

const testing::TestCase *testCaseUses(const testing::TestCase &Case) {
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:54: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const testing::TestSuite *testCaseUses(const testing::TestSuite &Case) {

  // No change for implicit declarations:
  auto Lambda = [&Case]() {};

  TestCaseInTemplate<testing::TestCase>();
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: TestCaseInTemplate<testing::TestSuite>();

  testing::TEST_CASE_CAN_FIX C1;
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestSuite C1;
  TEST_CASE_WARN_ONLY C2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'

  (void)new testing::TestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)new testing::TestSuite();
  const testing::TestCase *Result = &Case;
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const testing::TestSuite *Result = &Case;
  return Result;
}

struct TestCaseHolder {
  testing::TestCase Case;
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: testing::TestSuite Case;
};

class MyTest : public testing::TestCase {};
// CHECK-MESSAGES: [[@LINE-1]]:32: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class MyTest : public testing::TestSuite {};

template <typename T = testing::TestCase>
// CHECK-MESSAGES: [[@LINE-1]]:33: warning: Google Test APIs named with 'case'
// CHECK-FIXES: template <typename T = testing::TestSuite>
class TestTypeHolder {};

template <>
class TestTypeHolder<testing::TestCase> {};
// CHECK-MESSAGES: [[@LINE-1]]:31: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class TestTypeHolder<testing::TestSuite> {};

namespace shadow_using_ns {

using testing::TestCase;
// CHECK-MESSAGES: [[@LINE-1]]:16: warning: Google Test APIs named with 'case'
// CHECK-FIXES: using testing::TestSuite;

const TestCase *testCaseUses(const TestCase &Case) {
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:36: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const TestSuite *testCaseUses(const TestSuite &Case) {

  // No change for implicit declarations:
  auto Lambda = [&Case]() {};

  (void)new TestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)new TestSuite();
  const TestCase *Result = &Case;
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const TestSuite *Result = &Case;
  return Result;
}

struct TestCaseHolder {
  TestCase Case;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: TestSuite Case;
};

class MyTest : public TestCase {};
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class MyTest : public TestSuite {};

template <typename T = TestCase>
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: Google Test APIs named with 'case'
// CHECK-FIXES: template <typename T = TestSuite>
class TestTypeHolder {};

template <>
class TestTypeHolder<TestCase> {};
// CHECK-MESSAGES: [[@LINE-1]]:22: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class TestTypeHolder<TestSuite> {};

} // namespace shadow_using_ns

const shadow_using_ns::TestCase *shadowTestCaseUses(
    const shadow_using_ns::TestCase &Case) {
  // CHECK-MESSAGES: [[@LINE-2]]:24: warning: Google Test APIs named with 'case'
  // CHECK-MESSAGES: [[@LINE-2]]:28: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const shadow_using_ns::TestSuite *shadowTestCaseUses(
  // CHECK-FIXES: const shadow_using_ns::TestSuite &Case) {

  // No match for implicit declarations, as in the lambda capture:
  auto Lambda = [&Case]() {};

  (void)new shadow_using_ns::TestCase();
  // CHECK-MESSAGES: [[@LINE-1]]:30: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: (void)new shadow_using_ns::TestSuite();
  const shadow_using_ns::TestCase *Result = &Case;
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: const shadow_using_ns::TestSuite *Result = &Case;
  return Result;
}

struct ShadowTestCaseHolder {
  shadow_using_ns::TestCase Case;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: Google Test APIs named with 'case'
  // CHECK-FIXES: shadow_using_ns::TestSuite Case;
};

class ShadowMyTest : public shadow_using_ns::TestCase {};
// CHECK-MESSAGES: [[@LINE-1]]:46: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class ShadowMyTest : public shadow_using_ns::TestSuite {};

template <typename T = shadow_using_ns::TestCase>
// CHECK-MESSAGES: [[@LINE-1]]:41: warning: Google Test APIs named with 'case'
// CHECK-FIXES: template <typename T = shadow_using_ns::TestSuite>
class ShadowTestTypeHolder {};

template <>
class ShadowTestTypeHolder<shadow_using_ns::TestCase> {};
// CHECK-MESSAGES: [[@LINE-1]]:45: warning: Google Test APIs named with 'case'
// CHECK-FIXES: class ShadowTestTypeHolder<shadow_using_ns::TestSuite> {};

namespace typedef_ns {

typedef testing::TestCase MyTestCase;
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: Google Test APIs named with 'case'
// CHECK-FIXES: typedef testing::TestSuite MyTestCase;

const MyTestCase *testCaseUses(const MyTestCase &Case) {
  auto Lambda = [&Case]() {};
  (void)new MyTestCase();
  const MyTestCase *Result = &Case;
  return Result;
}

struct TestCaseHolder {
  MyTestCase Case;
};

class MyTest : public MyTestCase {};

template <typename T = MyTestCase>
class TestTypeHolder {};

template <>
class TestTypeHolder<MyTestCase> {};

} // namespace typedef_ns

const typedef_ns::MyTestCase *typedefTestCaseUses(
    const typedef_ns::MyTestCase &Case) {
  auto Lambda = [&Case]() {};
  (void)new typedef_ns::MyTestCase();
  const typedef_ns::MyTestCase *Result = &Case;
  return Result;
}

struct TypedefTestCaseHolder {
  typedef_ns::MyTestCase Case;
};

class TypedefMyTest : public typedef_ns::MyTestCase {};
template <typename T = typedef_ns::MyTestCase> class TypedefTestTypeHolder {};
template <> class TypedefTestTypeHolder<typedef_ns::MyTestCase> {};

namespace alias_ns {

using MyTestCase = testing::TestCase;
// CHECK-MESSAGES: [[@LINE-1]]:29: warning: Google Test APIs named with 'case'
// CHECK-FIXES: using MyTestCase = testing::TestSuite;

const MyTestCase *testCaseUses(const MyTestCase &Case) {
  auto Lambda = [&Case]() {};
  (void)new MyTestCase();
  const MyTestCase *Result = &Case;
  return Result;
}

struct TestCaseHolder {
  MyTestCase Case;
};

class MyTest : public MyTestCase {};
template <typename T = MyTestCase> class TestTypeHolder {};
template <> class TestTypeHolder<MyTestCase> {};

} // namespace alias_ns

const alias_ns::MyTestCase *aliasTestCaseUses(
    const alias_ns::MyTestCase &Case) {
  auto Lambda = [&Case]() {};
  (void)new alias_ns::MyTestCase();
  const alias_ns::MyTestCase *Result = &Case;
  return Result;
}

struct AliasTestCaseHolder {
  alias_ns::MyTestCase Case;
};

class AliasMyTest : public alias_ns::MyTestCase {};
template <typename T = alias_ns::MyTestCase> class AliasTestTypeHolder {};
template <> class AliasTestTypeHolder<alias_ns::MyTestCase> {};

template <typename T>
void templateFunction(const T& t) {
  (void)t.current_test_case();
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: Google Test APIs named with 'case'
}

void instantiateTemplateFunction(const testing::UnitTest &Test) {
  templateFunction(Test);
}
