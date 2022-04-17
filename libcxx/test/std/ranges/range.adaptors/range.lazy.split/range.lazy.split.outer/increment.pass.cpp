//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr outer-iterator& outer-iterator::operator++();
// constexpr decltype(auto) outer-iterator::operator++(int);

// Note that corner cases are tested in `range.lazy.split/general.pass.cpp`.

#include <ranges>

#include <algorithm>
#include <string_view>
#include "../small_string.h"
#include "../types.h"

constexpr bool test() {
  // Can call `outer-iterator::operator++`; `View` is a forward range.
  {
    SplitViewForward v("abc def ghi", " ");

    // ++i
    {
      auto i = v.begin();
      assert(*i == "abc"_str);

      decltype(auto) i2 = ++i;
      static_assert(std::is_lvalue_reference_v<decltype(i2)>);
      assert(&i2 == &i);
      assert(*i2 == "def"_str);
    }

    // i++
    {
      auto i = v.begin();
      assert(*i == "abc"_str);

      decltype(auto) i2 = i++;
      static_assert(!std::is_reference_v<decltype(i2)>);
      assert(*i2 == "abc"_str);
      assert(*i == "def"_str);
    }
  }

  // Can call `outer-iterator::operator++`; `View` is an input range.
  {
    SplitViewInput v("abc def ghi", ' ');

    // ++i
    {
      auto i = v.begin();
      assert(*i == "abc"_str);

      decltype(auto) i2 = ++i;
      static_assert(std::is_lvalue_reference_v<decltype(i2)>);
      assert(&i2 == &i);
      assert(*i2 == "def"_str);
    }

    // i++
    {
      auto i = v.begin();
      assert(*i == "abc"_str);

      static_assert(std::is_void_v<decltype(i++)>);
      i++;
      assert(*i == "def"_str);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
