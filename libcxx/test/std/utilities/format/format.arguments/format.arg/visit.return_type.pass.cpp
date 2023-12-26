//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// class basic_format_arg;

// template<class R, class Visitor>
//   R visit(this basic_format_arg arg, Visitor&& vis);

#include <algorithm>
#include <cassert>
#include <format>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")

template <class Context, class To, class ExpectedResult, class From>
void test(From value, ExpectedResult expectedValue) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  // member
  {
    std::same_as<ExpectedResult> decltype(auto) result =
        format_args.get(0).template visit<ExpectedResult>([v = To(value)](auto a) -> ExpectedResult {
          if constexpr (std::is_same_v<To, decltype(a)>) {
            assert(v == a);

            if constexpr (std::is_same_v<ExpectedResult, bool>) {
              return true;
            } else if constexpr (std::is_same_v<ExpectedResult, long>) {
              return 192812079084L;
            } else {
              return "visited";
            }
          } else {
            assert(false);
            return {};
          }
        });

    assert(result == expectedValue);
  }
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  std::basic_string<CharT> empty;
  std::basic_string<CharT> str = MAKE_STRING(CharT, "abc");

  // Test boolean types.

  test<Context, bool, std::string>(true, "visited");
  test<Context, bool, std::string>(false, "visited");

  test<Context, bool, bool>(true, true);
  test<Context, bool, bool>(false, true);

  test<Context, bool, long>(true, 192812079084L);
  test<Context, bool, long>(false, 192812079084L);

  // Test CharT types.

  test<Context, CharT, std::string, CharT>('a', "visited");
  test<Context, CharT, std::string, CharT>('z', "visited");
  test<Context, CharT, std::string, CharT>('0', "visited");
  test<Context, CharT, std::string, CharT>('9', "visited");
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
