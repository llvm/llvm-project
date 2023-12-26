//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// template<class Visitor, class Context>
//   see below visit_format_arg(Visitor&& vis, basic_format_arg<Context> arg);

#include <algorithm>
#include <cassert>
#include <format>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class Context, class To, class From>
void test(From value) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  // non-member
  {
    // expected-warning@+1 {{std::basic_format_context<char *, char>>' is deprecated}}
    std::visit_format_arg(
        [v = To(value)](auto a) -> To {
          if constexpr (std::is_same_v<To, decltype(a)>) {
            assert(v == a);
            return a;
          } else {
            assert(false);
            return {};
          }
        },
        format_args.get(0));
  }
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  std::basic_string<CharT> empty;
  std::basic_string<CharT> str = MAKE_STRING(CharT, "abc");

  test<Context, bool>(true);
}

int main(int, char**) {
  test<char>();

  return 0;
}
