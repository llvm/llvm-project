//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr outer-iterator::value-type outer-iterator::operator*() const;

#include <ranges>

#include <algorithm>
#include <cassert>
#include <string>
#include <string_view>
#include <type_traits>
#include "../types.h"

template <class View, class Separator>
constexpr void test_one(Separator sep) {
  using namespace std::string_literals;
  using namespace std::string_view_literals;

  View v("abc def ghi"sv, sep);

  // Non-const iterator.
  {
    auto i = v.begin();
    static_assert(!std::is_reference_v<decltype(*i)>);
    assert(std::ranges::equal(*i, "abc"s));
    assert(std::ranges::equal(*(++i), "def"s));
    assert(std::ranges::equal(*(++i), "ghi"s));
  }

  // Const iterator.
  {
    const auto ci = v.begin();
    static_assert(!std::is_reference_v<decltype(*ci)>);
    assert(std::ranges::equal(*ci, "abc"s));
  }
}

constexpr bool test() {
  // `View` is a forward range.
  test_one<SplitViewDiff>(" ");
  test_one<SplitViewInput>(' ');

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
