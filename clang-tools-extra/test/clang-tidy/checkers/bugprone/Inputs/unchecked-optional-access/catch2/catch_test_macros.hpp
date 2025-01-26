#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_CATCH2_CATCH_TEST_MACROS_HPP_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_CATCH2_CATCH_TEST_MACROS_HPP_

// Mock version of catch2 test framework macros.

// The macros are normally much more complex. For
// - REQUIRE and REQUIRE_FALSE: define them to go through some wrapper
//   (that is not at all what catch2 actually does)
// - TEST_CASE and METHOD_AS_TEST_CASE: define a function
void Wrapper(bool cond) {
  if (!cond)
    __builtin_trap();
}

#define CONCAT_HELP(X,Y) X##Y  // helper macro
#define CONCAT(X,Y) CONCAT_HELP(X,Y)

// Catch2 can be configured to have a prefix "CATCH" for macro names, or not,
// so we model this.
#ifdef CATCH_CONFIG_PREFIX_ALL
#define CATCH_REQUIRE( ... ) Wrapper(static_cast<bool>(__VA_ARGS__))
#define CATCH_REQUIRE_FALSE( ... ) Wrapper(static_cast<bool>(__VA_ARGS__))
#define CATCH_TEST_CASE( ... ) void CONCAT(TEST, __LINE__)()
#define CATCH_METHOD_AS_TEST_CASE( ... ) void CONCAT(TEST, __LINE__)()
#else
#define REQUIRE( ... ) Wrapper(static_cast<bool>(__VA_ARGS__))
#define REQUIRE_FALSE( ... ) Wrapper(static_cast<bool>(__VA_ARGS__))
#define TEST_CASE( ... ) void CONCAT(TEST, __LINE__)()
#define METHOD_AS_TEST_CASE( ... ) void CONCAT(TEST, __LINE__)()
#endif

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_CATCH2_CATCH_TEST_MACROS_HPP_
