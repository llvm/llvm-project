// RUN: %check_clang_tidy %s google-readability-avoid-underscore-in-googletest-name %t

#define TEST(test_case_name, test_name) void test_case_name##test_name()
#define TEST_F(test_case_name, test_name) void test_case_name##test_name()
#define TEST_P(test_case_name, test_name) void test_case_name##test_name()
#define TYPED_TEST(test_case_name, test_name) void test_case_name##test_name()
#define TYPED_TEST_P(test_case_name, test_name) void test_case_name##test_name()
#define FRIEND_TEST(test_case_name, test_name) void test_case_name##test_name()

TEST(TestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST(TestCaseName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(TestCaseName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_TestCaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test case name "Illegal_TestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_Test_CaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test case name "Illegal_Test_CaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST(Illegal_TestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using "_" in test case name "Illegal_TestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:28: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(TestCaseFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(TestCaseFixtureName, DISABLED_Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(TestCaseFixtureName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(Illegal_TestCaseFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_TestCaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_F(Illegal_TestCaseFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_TestCaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:37: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_F(Illegal_Test_CaseFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_Test_CaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(ParameterizedTestCaseFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(ParameterizedTestCaseFixtureName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(ParameterizedTestCaseFixtureName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(Illegal_ParameterizedTestCaseFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_ParameterizedTestCaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TEST_P(Illegal_ParameterizedTestCaseFixtureName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_ParameterizedTestCaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:50: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TEST_P(Illegal_Parameterized_TestCaseFixtureName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using "_" in test case name "Illegal_Parameterized_TestCaseFixtureName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(TypedTestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(TypedTestCaseName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(TypedTestCaseName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(Illegal_TypedTestCaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test case name "Illegal_TypedTestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST(Illegal_TypedTestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test case name "Illegal_TypedTestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:39: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST(Illegal_Typed_TestCaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using "_" in test case name "Illegal_Typed_TestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(TypeParameterizedTestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(TypeParameterizedTestCaseName, DISABLED_Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(TypeParameterizedTestCaseName, Illegal_Test_Name) {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: avoid using "_" in test name "Illegal_Test_Name" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(Illegal_TypeParameterizedTestCaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test case name "Illegal_TypeParameterizedTestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
TYPED_TEST_P(Illegal_TypeParameterizedTestCaseName, Illegal_TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test case name "Illegal_TypeParameterizedTestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]
// CHECK-MESSAGES: :[[@LINE-2]]:53: warning: avoid using "_" in test name "Illegal_TestName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

TYPED_TEST_P(Illegal_Type_ParameterizedTestCaseName, TestName) {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using "_" in test case name "Illegal_Type_ParameterizedTestCaseName" according to Googletest FAQ [google-readability-avoid-underscore-in-googletest-name]

// Underscores are allowed to disable a test with the DISABLED_ prefix.
// https://github.com/google/googletest/blob/master/googletest/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore
TEST(TestCaseName, TestName) {}
TEST(TestCaseName, DISABLED_TestName) {}

TEST_F(TestCaseFixtureName, TestName) {}
TEST_F(TestCaseFixtureName, DISABLED_TestName) {}

TEST_P(ParameterizedTestCaseFixtureName, TestName) {}
TEST_P(ParameterizedTestCaseFixtureName, DISABLED_TestName) {}

TYPED_TEST(TypedTestName, TestName) {}
TYPED_TEST(TypedTestName, DISABLED_TestName) {}

TYPED_TEST_P(TypeParameterizedTestName, TestName) {}
TYPED_TEST_P(TypeParameterizedTestName, DISABLED_TestName) {}

FRIEND_TEST(FriendTest, Is_NotChecked) {}
FRIEND_TEST(Friend_Test, IsNotChecked) {}
FRIEND_TEST(Friend_Test, Is_NotChecked) {}
