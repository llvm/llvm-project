// RUN: %check_clang_tidy %s google-readability-avoid-underscore-in-googletest-name %t

#define TEST(test_suite_name, test_name) void test_suite_name##test_name()
#define TEST_F(test_suite_name, test_name) void test_suite_name##test_name()
#define TEST_P(test_suite_name, test_name) void test_suite_name##test_name()
#define TYPED_TEST(test_suite_name, test_name) void test_suite_name##test_name()
#define TYPED_TEST_P(test_suite_name, test_name) void test_suite_name##test_name()
#define FRIEND_TEST(test_suite_name, test_name) void test_suite_name##test_name()

TEST(TestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST(TestSuiteName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(TestSuiteName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_TestSuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test suite name "Illegal_TestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_Test_SuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test suite name "Illegal_Test_SuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_TestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test suite name "Illegal_TestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:29: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(TestSuiteFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(TestSuiteFixtureName, DISABLED_Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(TestSuiteFixtureName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(Illegal_TestSuiteFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_TestSuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(Illegal_TestSuiteFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_TestSuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:38: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(Illegal_Test_SuiteFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_Test_SuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(ParameterizedTestSuiteFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:43: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(ParameterizedTestSuiteFixtureName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:43: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(ParameterizedTestSuiteFixtureName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:43: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(Illegal_ParameterizedTestSuiteFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_ParameterizedTestSuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(Illegal_ParameterizedTestSuiteFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_ParameterizedTestSuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:51: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(Illegal_Parameterized_TestSuiteFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test suite name "Illegal_Parameterized_TestSuiteFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(TypedTestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(TypedTestSuiteName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(TypedTestSuiteName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(Illegal_TypedTestSuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test suite name "Illegal_TypedTestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(Illegal_TypedTestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test suite name "Illegal_TypedTestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:40: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(Illegal_Typed_TestSuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test suite name "Illegal_Typed_TestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(TypeParameterizedTestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(TypeParameterizedTestSuiteName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(TypeParameterizedTestSuiteName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(Illegal_TypeParameterizedTestSuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test suite name "Illegal_TypeParameterizedTestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(Illegal_TypeParameterizedTestSuiteName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test suite name "Illegal_TypeParameterizedTestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:54: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(Illegal_Type_ParameterizedTestSuiteName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test suite name "Illegal_Type_ParameterizedTestSuiteName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

// Underscores are allowed to disable a test with the DISABLED_ prefix.
// https://google.github.io/googletest/faq.html#why-should-test-suite-names-and-test-names-not-contain-underscore
TEST(TestSuiteName, TestName) {}
TEST(TestSuiteName, DISABLED_TestName) {}
TEST(DISABLED_TestSuiteName, TestName) {}
TEST(DISABLED_TestSuiteName, DISABLED_TestName) {}

TEST_F(TestSuiteFixtureName, TestName) {}
TEST_F(TestSuiteFixtureName, DISABLED_TestName) {}
TEST_F(DISABLED_TestSuiteFixtureName, TestName) {}
TEST_F(DISABLED_TestSuiteFixtureName, DISABLED_TestName) {}

TEST_P(ParameterizedTestSuiteFixtureName, TestName) {}
TEST_P(ParameterizedTestSuiteFixtureName, DISABLED_TestName) {}
TEST_P(DISABLED_ParameterizedTestSuiteFixtureName, TestName) {}
TEST_P(DISABLED_ParameterizedTestSuiteFixtureName, DISABLED_TestName) {}

TYPED_TEST(TypedTestSuiteName, TestName) {}
TYPED_TEST(TypedTestSuiteName, DISABLED_TestName) {}
TYPED_TEST(DISABLED_TypedTestSuiteName, TestName) {}
TYPED_TEST(DISABLED_TypedTestSuiteName, DISABLED_TestName) {}

TYPED_TEST_P(TypeParameterizedTestSuiteName, TestName) {}
TYPED_TEST_P(TypeParameterizedTestSuiteName, DISABLED_TestName) {}
TYPED_TEST_P(DISABLED_TypeParameterizedTestSuiteName, TestName) {}
TYPED_TEST_P(DISABLED_TypeParameterizedTestSuiteName, DISABLED_TestName) {}

FRIEND_TEST(FriendTestSuite, Is_NotChecked) {}
FRIEND_TEST(Friend_TestSuite, IsNotChecked) {}
FRIEND_TEST(Friend_TestSuite, Is_NotChecked) {}
