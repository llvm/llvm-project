//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr outer-iterator& outer-iterator::operator++();
// constexpr decltype(auto) outer-iterator::operator++(int);

// Note that corner cases are tested in `range.lazy.split/general.pass.cpp`.

#include <ranges>

#include <algorithm>
#include <cassert>
#include <string>
#include "../types.h"

constexpr bool test() {
  using namespace std::string_literals;
  // Can call `outer-iterator::operator++`; `View` is a forward range.
  {
    SplitViewForward v("abc def ghi", " ");

    // ++i
    {
      auto i = v.begin();
      assert(std::ranges::equal(*i, "abc"s));

      decltype(auto) i2 = ++i;
      static_assert(std::is_lvalue_reference_v<decltype(i2)>);
      assert(&i2 == &i);
      assert(std::ranges::equal(*i2, "def"s));
    }

    // i++
    {
      auto i = v.begin();
      assert(std::ranges::equal(*i, "abc"s));

      decltype(auto) i2 = i++;
      static_assert(!std::is_reference_v<decltype(i2)>);
      assert(std::ranges::equal(*i2, "abc"s));
      assert(std::ranges::equal(*i, "def"s));
    }
  }

  // Can call `outer-iterator::operator++`; `View` is an input range.
  {
    SplitViewInput v("abc def ghi", ' ');

    // ++i
    {
      auto i = v.begin();
      assert(std::ranges::equal(*i, "abc"s));

      decltype(auto) i2 = ++i;
      static_assert(std::is_lvalue_reference_v<decltype(i2)>);
      assert(&i2 == &i);
      assert(std::ranges::equal(*i2, "def"s));
    }

    // i++
    {
      auto i = v.begin();
      assert(std::ranges::equal(*i, "abc"s));

      static_assert(std::is_void_v<decltype(i++)>);
      i++;
      assert(std::ranges::equal(*i, "def"s));
    }
  }

  // LWG3505
  {
    using namespace std::string_view_literals;

    { // Motivational example
      auto v = std::views::lazy_split("xxyx"sv, "xy"sv);

      {
        auto i = v.begin();
        assert(std::ranges::equal(*i, "x"s));

        decltype(auto) i2 = ++i;
        static_assert(std::is_lvalue_reference_v<decltype(i2)>);
        assert(std::ranges::equal(*i2, "x"s));
      }

      {
        auto i = v.begin();
        assert(std::ranges::equal(*i, "x"s));

        decltype(auto) i2 = i++;
        static_assert(!std::is_reference_v<decltype(i2)>);
        assert(std::ranges::equal(*i2, "x"s));
        assert(std::ranges::equal(*i, "x"s));
      }
    }
    {
      auto v = std::views::lazy_split("zzht"sv, "zh"sv);

      {
        auto i = v.begin();
        assert(std::ranges::equal(*i, "z"s));

        decltype(auto) i2 = ++i;
        static_assert(std::is_lvalue_reference_v<decltype(i2)>);
        assert(std::ranges::equal(*i2, "t"s));
      }

      {
        auto i = v.begin();
        assert(std::ranges::equal(*i, "z"s));

        decltype(auto) i2 = i++;
        static_assert(!std::is_reference_v<decltype(i2)>);
        assert(std::ranges::equal(*i2, "z"s));
        assert(std::ranges::equal(*i, "t"s));
      }
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
