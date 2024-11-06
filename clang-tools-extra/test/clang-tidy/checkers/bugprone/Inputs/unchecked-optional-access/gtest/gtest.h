#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_

// Mock version of googletest macros.

// Normally this declares a class, but it isn't relevant for testing.
#define GTEST_TEST(test_suite_name, test_name)

// Normally, this has a relatively complex implementation
// (wrapping the condition evaluation), more complex failure behavior, etc.,
// but we keep it simple for testing.
#define ASSERT_TRUE(condition)                                                 \
  if (condition)                                                               \
    ;                                                                          \
  else                                                                         \
    return;  // normally "fails" rather than just return

#endif  // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
