//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// template <class T> class optional::iterator;
// template <class T> class optional::const_iterator;

#include <cassert>
#include <compare>
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

template <typename T>
constexpr bool test() {
  using Opt = std::optional<T>;
  using I   = Opt::iterator;
  using CI  = Opt::const_iterator;

  static_assert(std::three_way_comparable<I>);
  static_assert(std::three_way_comparable<CI>);

  std::remove_reference_t<T> t{};
  Opt opt{t};

  // [container.reqmts] tests for comparison operators of optional::iterator and optional::const_iterator
  auto it1 = opt.begin();

  {
    auto it2 = opt.begin();
    assert(it1 == it2);
    assert(!(it1 != it2));

    static_assert(std::same_as<decltype(it1 <=> it2), std::strong_ordering>);
    assert(it1 <=> it2 == std::strong_ordering::equal);
  }

  {
    auto it3 = opt.end();
    assert(it1 != it3);
    assert(it1 <= it3);
    assert(it1 < it3);
    assert(it3 >= it1);
    assert(it3 > it1);

    assert(it1 <=> it3 == std::strong_ordering::less);
    assert(it3 <=> it1 == std::strong_ordering::greater);
  }

  auto cit1 = std::as_const(opt).begin();

  {
    auto cit2 = std::as_const(opt).begin();
    assert(cit1 == cit2);
    assert(!(cit1 != cit2));

    static_assert(std::same_as<decltype(cit1 <=> cit2), std::strong_ordering>);
    assert(cit1 <=> cit2 == std::strong_ordering::equal);
  }

  {
    auto cit3 = std::as_const(opt).end();

    assert(cit1 <= cit3);
    assert(cit1 < cit3);
    assert(cit3 >= cit1);
    assert(cit3 > cit1);

    assert(cit1 <=> cit3 == std::strong_ordering::less);
    assert(cit3 <=> cit1 == std::strong_ordering::greater);
  }

  return true;
}

constexpr bool test() {
  test<int>();
  test<char>();
  test<int&>();

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
