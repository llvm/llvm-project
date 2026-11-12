#ifndef THIRD_PARTY_LLVM_LLVM_TOOLS_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_INPUTS_GTEST_GTEST_H_
#define THIRD_PARTY_LLVM_LLVM_TOOLS_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_INPUTS_GTEST_GTEST_H_

#include "gtest/gtest-typed-test.h"

namespace testing {

class Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  static void SetUpTestSuite();
  static void TearDownTestSuite();
};

class TestSuite {};
using TestCase = TestSuite;

class TestInfo {
public:
  const char *test_case_name() const;

  const char *test_suite_name() const;
};

class TestEventListener {
public:
  virtual void OnTestCaseStart(const TestCase &);
  virtual void OnTestCaseEnd(const TestCase &);

  virtual void OnTestSuiteStart(const TestCase &);
  virtual void OnTestSuiteEnd(const TestCase &);
};

class EmptyTestEventListener : public TestEventListener {
public:
  void OnTestCaseStart(const TestCase &) override;
  void OnTestCaseEnd(const TestCase &) override;

  void OnTestSuiteStart(const TestCase &) override;
  void OnTestSuiteEnd(const TestCase &) override;
};

class UnitTest {
public:
  static UnitTest *GetInstance();

  TestCase *current_test_case() const;
  int successful_test_case_count() const;
  int failed_test_case_count() const;
  int total_test_case_count() const;
  int test_case_to_run_count() const;
  const TestCase *GetTestCase(int) const;

  TestSuite *current_test_suite() const;
  int successful_test_suite_count() const;
  int failed_test_suite_count() const;
  int total_test_suite_count() const;
  int test_suite_to_run_count() const;
  const TestSuite *GetTestSuite(int) const;
};

} // namespace testing

#endif // THIRD_PARTY_LLVM_LLVM_TOOLS_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_INPUTS_GTEST_GTEST_H_
