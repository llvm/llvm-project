//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr outer-iterator::value-type outer-iterator::operator*() const;

#include <ranges>

#include <cassert>
#include <string_view>
#include <type_traits>
#include "../small_string.h"
#include "../types.h"

template <class View, class Separator>
constexpr void test_one(Separator sep) {
  using namespace std::string_view_literals;

  View v("abc def ghi"sv, sep);

  // Non-const iterator.
  {
    auto i = v.begin();
    static_assert(!std::is_reference_v<decltype(*i)>);
    assert(SmallString(*i) == "abc"_str);
    assert(SmallString(*(++i)) == "def"_str);
    assert(SmallString(*(++i)) == "ghi"_str);
  }

  // Const iterator.
  {
    const auto ci = v.begin();
    static_assert(!std::is_reference_v<decltype(*ci)>);
    assert(SmallString(*ci) == "abc"_str);
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
