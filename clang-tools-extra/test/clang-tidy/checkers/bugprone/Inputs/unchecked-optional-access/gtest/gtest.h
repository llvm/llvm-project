#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_

// Mock version of googletest macros.

#define GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)                     \
  test_suite_name##_##test_name##_Test

#define GTEST_TEST(test_suite_name, test_name)                                 \
  class GTEST_TEST_CLASS_NAME_(test_suite_name, test_name) {                   \
  public:                                                                      \
    GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() = default;            \
  };

#define GTEST_AMBIGUOUS_ELSE_BLOCKER_                                          \
  switch (0)                                                                   \
  case 0:                                                                      \
  default: // NOLINT

#define ASSERT_TRUE(condition)                                                 \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
  if (condition)                                                               \
    ;                                                                          \
  else                                                                         \
    return;  // should fail...

#endif  // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
