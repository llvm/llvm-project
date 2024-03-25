//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing

// <format>

// [format.string.std]/8
// If { arg-idopt } is used in a width or precision, the value of the
// corresponding formatting argument is used in its place. If the
// corresponding formatting argument is not of standard signed or unsigned
// integer type, or its value is negative for precision or non-positive for
// width, an exception of type format_error is thrown.
//
// This test does the run-time validation

#include <cassert>
#include <format>

#include "assert_macros.h"
#include "concat_macros.h"
#include "format.functions.common.h"
#include "make_string.h"
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class... Args>
void test_exception([[maybe_unused]] std::basic_string_view<CharT> fmt, [[maybe_unused]] Args&&... args) {
  [[maybe_unused]] std::string_view what = "Replacement argument isn't a standard signed or unsigned integer type";
  TEST_VALIDATE_EXCEPTION(
      std::format_error,
      [&]([[maybe_unused]] const std::format_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED(
                "\nFormat string   ", fmt, "\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::vformat(fmt, std::make_format_args<context_t<CharT>>(args...)));
}

template <class CharT>
void test() {
  // *** Width ***
  test_exception(SV("{:{}}"), 42, true);
  test_exception(SV("{:{}}"), 42, '0');
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (std::same_as<CharT, wchar_t>)
    test_exception(SV("{:{}}"), 42, L'0');
#endif
#ifndef TEST_HAS_NO_INT128
  test_exception(SV("{:{}}"), 42, __int128_t(0));
  test_exception(SV("{:{}}"), 42, __uint128_t(0));
#endif
  test_exception(SV("{:{}}"), 42, 42.0f);
  test_exception(SV("{:{}}"), 42, 42.0);
  test_exception(SV("{:{}}"), 42, 42.0l);

  // *** Precision ***
  test_exception(SV("{:0.{}}"), 42.0, true);
  test_exception(SV("{:0.{}}"), 42.0, '0');
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (std::same_as<CharT, wchar_t>)
    test_exception(SV("{:0.{}}"), 42.0, L'0');
#endif
#ifndef TEST_HAS_NO_INT128
  test_exception(SV("{:0.{}}"), 42.0, __int128_t(0));
  test_exception(SV("{:0.{}}"), 42.0, __uint128_t(0));
#endif
  test_exception(SV("{:0.{}}"), 42.0, 42.0f);
  test_exception(SV("{:0.{}}"), 42.0, 42.0);
  test_exception(SV("{:0.{}}"), 42.0, 42.0l);
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
