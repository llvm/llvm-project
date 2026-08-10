//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME


// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG_BASIC_FORMAT_STRING_CACHE

// <format>


// This is a test for the new caching mechanism.

#include <format>
#include <cassert>
#include <vector>

#include "make_string.h"
#include "test_format_string.h"

#include <iostream>
#include <type_traits>

#define SV(S) MAKE_STRING_VIEW(CharT, S)
template < class CharT, class... Args>
void check(std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
#ifndef TEST_HAS_NO_LOCALIZATION
  if constexpr (std::same_as<CharT, char>)
    if (out != expected)
      std::cerr << "\nFormat string   " << fmt.get() << "\nExpected output " << expected << "\nActual output   " << out
                << '\n';
#endif
  assert(out == expected);
};

template <class CharT>
struct String {
  std::basic_string<CharT> s;
};

template <class CharT>
struct std::formatter<String<CharT>, CharT> : std::formatter<std::basic_string_view<CharT>> {
  template <class FormatContext>
  typename FormatContext::iterator format(const String<CharT>& str, FormatContext& ctx) const {
    return std::formatter<std::basic_string_view<CharT>>::format(std::basic_string_view<CharT >{str.s}, ctx);
  }
};

template <class CharT>
static void test() {
  check(SV("FULL:{"), SV("{{"));
  check(SV("FULL:foo{"), SV("foo{{"));
  check(SV("FULL:foo{{"), SV("foo{{{{"));

  check(SV("FULL:}"), SV("}}"));
  check(SV("FULL:foo}"), SV("foo}}"));
  check(SV("FULL:foo}}"), SV("foo}}}}"));

  check(SV("FULL:foo{}"), SV("foo{{}}"));
  check(SV("FULL:foo{{}}"), SV("foo{{{{}}}}"));
  check(SV("FULL:foo}{"), SV("foo}}{{"));
  check(SV("FULL:foo}}{{"), SV("foo}}}}{{{{"));

  check(SV("FULL:ZZZ"), SV("ZZZ"));
  check(SV("FULL:Z{Z"), SV("Z{{Z"));
  check(SV("FULL:Z}Z"), SV("Z}}Z"));
  check(SV("FULL:Z{Z{"), SV("Z{{Z{{"));
  check(SV("FULL:Z}Z{Z"), SV("Z}}Z{{Z"));

  check(SV("FULL:ZZZ"), SV("{}"), SV("ZZZ"));
  check(SV("FULL:42"), SV("{}"), 42);
  check(SV("FULL:true"), SV("{}"), true);

  check(SV("FULL:ZZZ"), SV("{:}"), SV("ZZZ"));
  check(SV("FULL:0x0042"), SV("{:#06x}"), 0x42);
  check(SV("FULL:0x0042=answer"), SV("{:#06x}={}"), 0x42, SV("answer"));

//  check(SV("FULL:hello world"), SV("{} world"), String<char>{"hello"});
//  check(SV("FULL:hello world"), SV("{0:} world"), String<char>{"hello"});

  // TODO TEST WITH ARG EATER
}

int main(int, char**) {
  test<char>();

  return 0;
}
