//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <thread>

// template<class charT>
// struct formatter<thread::id, charT>;

// template<class... Args>
//   string format(format_string<Args...> fmt, Args&&... args);
// template<class... Args>
//   wstring format(wformat_string<Args...> fmt, Args&&... args);

#include <cassert>
#include <format>
#include <thread>

#include "assert_macros.h"
#include "concat_macros.h"
#include "format.functions.common.h"
#include "test_format_string.h"
#include "test_macros.h"

template <class CharT, class TestFunction>
void format_tests(TestFunction check) {
  // Note the output of std::thread::id is unspecified. The output text is the
  // same as the stream operator. Since that format is already released this
  // test follows the practice on existing systems.
  std::thread::id input{};

  /***** Test the type specific part *****/
  check(SV("0"), SV("{}"), input);
  check(SV("0^42"), SV("{}^42"), input);
  check(SV("0^42"), SV("{:}^42"), input);

  // *** align-fill & width ***
  check(SV("    0"), SV("{:5}"), input);
  check(SV("0****"), SV("{:*<5}"), input);
  check(SV("__0__"), SV("{:_^5}"), input);
  check(SV("::::0"), SV("{::>5}"), input); // This is not a range, so : is allowed as fill character.

  check(SV("    0"), SV("{:{}}"), input, 5);
  check(SV("0****"), SV("{:*<{}}"), input, 5);
  check(SV("__0__"), SV("{:_^{}}"), input, 5);
  check(SV("####0"), SV("{:#>{}}"), input, 5);
}

auto test = []<class CharT, class... Args>(
                std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  {
    std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
  {
    std::basic_string<CharT> out = std::vformat(fmt.get(), std::make_format_args<context_t<CharT>>(args...));
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
};

int main(int, char**) {
  format_tests<char>(test);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests<wchar_t>(test);
#endif

  { // Check that we print the ID correctly (i.e. library-internal round-tripping works)
    auto thread_id = std::this_thread::get_id();
    auto result    = std::format("{}", thread_id);
    auto num       = std::stoull(result);
    using int_t =
        std::conditional_t<std::is_pointer<std::__libcpp_thread_id>::value, uintptr_t, std::__libcpp_thread_id>;
    assert(reinterpret_cast<int_t>(__get_underlying_id(std::this_thread::get_id())) == static_cast<int_t>(num));
  }

  return 0;
}
