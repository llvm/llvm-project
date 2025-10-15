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
#include <iterator>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>

template <typename T, T __val>
constexpr bool test() {
  std::optional<T> opt{__val};

  { // Dereferencing an iterator of an engaged optional will return the same value that the optional holds.
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    assert(*it == *opt);
    assert(*it2 == *std::as_const(opt));
  }

  { // optional::iterator and optional::const_iterator satisfy the Cpp17RandomAccessIterator and contiguous iterator.
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    assert(std::contiguous_iterator<decltype(it)>);
    assert(std::contiguous_iterator<decltype(it2)>);

    assert(std::random_access_iterator<decltype(it)>);
    assert(std::random_access_iterator<decltype(it2)>);
  }

  { // const_iterator::value_type == std::remove_cv_t<T>, const_iterator::reference == const T&, iterator::value_type = std::remove_cv_t<T>, iterator::reference == T&
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    assert((std::is_same_v<typename decltype(it)::value_type, std::remove_cv_t<T>>));
    assert((std::is_same_v<typename decltype(it)::reference, T&>));
    assert((std::is_same_v<typename decltype(it2)::value_type, std::remove_cv_t<T>>));
    assert((std::is_same_v<typename decltype(it2)::reference, const T&>));
  }

  { // std::ranges::size for an engaged optional<T> == 1, disengaged optional<T> == 0
    const std::optional<T> disengaged{std::nullopt};
    std::optional<T> disengaged2{std::nullopt};
    assert(std::ranges::size(opt) == 1);
    assert(std::ranges::size(std::as_const(opt)) == 1);

    assert(std::ranges::size(disengaged) == 0);
    assert(std::ranges::size(disengaged2) == 0);
  }

  { // std::ranges::enable_view<optional<T>> == true, and std::format_kind<optional<T>> == true
    static_assert(std::ranges::enable_view<std::optional<T>> == true);
    static_assert(std::format_kind<std::optional<T>> == std::range_format::disabled);
  }

  // An optional with value that is reset will have a begin() == end(), then when it is reassigned a value,
  // begin() != end(), and *begin() will contain the new value.
  {
    std::optional<T> val{__val};
    assert(val.begin() != val.end());
    val.reset();
    assert(val.begin() == val.end());
    val.emplace(__val);
    assert(val.begin() != val.end());
    assert(*(val.begin()) == __val);
  }

  return true;
}

constexpr bool tests() {
  assert((test<int, 1>()));
  assert((test<char, 'a'>()));
  assert((test<bool, true>()));
  assert((test<const int, 2>()));
  assert((test<const char, 'b'>()));

  return true;
}

int main(int, char**) {
  assert(tests());
  static_assert(tests());

  return 0;
}
