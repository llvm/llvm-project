//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const W & operator*() const noexcept;

#include <ranges>
#include <cassert>
#include <concepts>

constexpr bool test() {
  // unbound
  {
    std::ranges::repeat_view<int> v(31);
    auto iter = v.begin();

    const int& val = *iter;
    for (int i = 0; i < 100; ++i, ++iter) {
      assert(*iter == 31);
      assert(&*iter == &val);
    }

    static_assert(noexcept(*iter));
    static_assert(std::same_as<decltype(*iter), const int&>);
  }

  // bound && one element
  {
    std::ranges::repeat_view<int, int> v(31, 1);
    auto iter = v.begin();
    assert(*iter == 31);
    static_assert(noexcept(*iter));
    static_assert(std::same_as<decltype(*iter), const int&>);
  }

  // bound && several elements
  {
    std::ranges::repeat_view<int, int> v(31, 100);
    auto iter = v.begin();

    const int& val = *iter;
    for (int i = 0; i < 100; ++i, ++iter) {
      assert(*iter == 31);
      assert(&*iter == &val);
    }
  }

  // bound && foreach
  {
    for (const auto& val : std::views::repeat(31, 100))
      assert(val == 31);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
