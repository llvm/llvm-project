//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_ASSERT_MACROS_H
#define TEST_SUPPORT_ASSERT_MACROS_H

// Contains a set of validation macros.
//
// Note these test were added after C++20 was well supported by the compilers
// used. To make the implementation simple the macros require C++20 or newer.
// It's not expected that existing tests start to use these new macros.
//
// These macros are an alternative to using assert. The differences are:
// - The assert message isn't localized.
// - It's possible to log additional information. This is useful when the
//   function asserting is a helper function. In these cases the assertion
//   failure contains to little information to find the issue. For example, in
//   the format functions, the really useful information is the actual output,
//   the expected output, and the format string used. These macros allow
//   logging additional arguments.

#include "test_macros.h"

#include <cstdio>
#include <cstdlib>

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <sstream>
#endif

#if TEST_STD_VER > 17

#  ifndef TEST_HAS_NO_LOCALIZATION
template <class T>
concept test_char_streamable = requires(T&& value) { std::stringstream{} << std::forward<T>(value); };
#  endif

// If possible concatenates message for the assertion function, else returns a
// default message. Not being able to stream is not considered and error. For
// example, streaming to std::wcerr doesn't work properly in the CI. Therefore
// the formatting tests should only stream to std::string_string.
template <class... Args>
std::string test_concat_message([[maybe_unused]] Args&&... args) {
#  ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr ((test_char_streamable<Args> && ...)) {
    std::stringstream sstr;
    ((sstr << std::forward<Args>(args)), ...);
    return sstr.str();
  } else
#  endif
    return "Message discarded since it can't be streamed to std::cerr.\n";
}

#endif // TEST_STD_VER > 17

// Logs the error and calls exit.
//
// It shows a generic assert like message including a custom message. This
// message should end with a newline.
[[noreturn]] void test_log_error(const char* condition, const char* file, int line, std::string&& message) {
  const char* msg = condition ? "Assertion failure: " : "Unconditional failure:";
  std::fprintf(stderr, "%s%s %s %d\n%s", msg, condition, file, line, message.c_str());
  exit(EXIT_FAILURE);
}

inline void test_fail(const char* file, int line, std::string&& message) {
  test_log_error("", file, line, std::move(message));
}

inline void test_require(bool condition, const char* condition_str, const char* file, int line, std::string&& message) {
  if (condition)
    return;

  test_log_error(condition_str, file, line, std::move(message));
}

inline void test_libcpp_require(
    [[maybe_unused]] bool condition,
    [[maybe_unused]] const char* condition_str,
    [[maybe_unused]] const char* file,
    [[maybe_unused]] int line,
    [[maybe_unused]] std::string&& message) {
#if defined(_LIBCPP_VERSION)
  test_require(condition, condition_str, file, line, std::move(message));
#endif
}

// assert(false) replacement
#define TEST_FAIL(MSG) ::test_fail(__FILE__, __LINE__, MSG)

// assert replacement.
#define TEST_REQUIRE(CONDITION, MSG) ::test_require(CONDITION, #CONDITION, __FILE__, __LINE__, MSG)

// LIBCPP_ASSERT replacement
//
// This requirement is only tested when the test suite is used for libc++.
// This allows checking libc++ specific requirements, for example the error
// messages of exceptions.
#define TEST_LIBCPP_REQUIRE(CONDITION, MSG) ::test_libcpp_require(CONDITION, #CONDITION, __FILE__, __LINE__, MSG)

#endif // TEST_SUPPORT_ASSERT_MACROS_H
