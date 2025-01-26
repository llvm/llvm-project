#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_

// Mock version of googletest macros.

// Normally this declares a class, but it isn't relevant for testing.
#define GTEST_TEST(test_suite_name, test_name)

// Normally googletest creates a class wrapping the condition, which the
// dataflow analysis framework has difficulty "seeing through"
// (needs some inter-procedural analysis, or special-case handling).
// Here is a simplified version.
class BoolWrapper {
public:
  template <typename T>
  BoolWrapper(T &&val) : success_(static_cast<bool>(val)) {}
  operator bool() const { return success_; }
  bool success_;
};

#define ASSERT_TRUE(condition)                                                 \
  if (BoolWrapper(condition))                                                  \
    ;                                                                          \
  else                                                                         \
    return;

#define ASSERT_FALSE(condition)                                                \
  if (BoolWrapper(!condition))                                                 \
    ;                                                                          \
  else                                                                         \
    return;

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_GTEST_GTEST_H_
