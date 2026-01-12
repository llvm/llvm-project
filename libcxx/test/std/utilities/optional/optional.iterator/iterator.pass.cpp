//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: libcpp-has-no-experimental-optional-iterator

// <optional>

// template <class T> class optional::iterator;
// template <class T> class optional::const_iterator;

#include <cassert>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>

template <typename T>
concept has_iterator = requires { typename T::iterator; };

template <typename T>
concept has_const_iterator = requires { typename T::const_iterator; };

template <typename T>
concept has_both_iterators = has_iterator<T> && has_const_iterator<T>;

template <typename T>
concept only_has_iterator = has_iterator<T> && !has_const_iterator<T>;

template <typename T>
concept has_no_iterators = !has_iterator<T> && !has_const_iterator<T>;

template <typename T>
constexpr void test(std::decay_t<T> v) {
  std::optional<T> opt{v};
  {
    static_assert(std::ranges::range<decltype(opt)>);
  }

  { // Dereferencing an iterator of an engaged optional will return the same value that the optional holds.
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    assert(*it == *opt);
    assert(*it2 == *std::as_const(opt));
  }

  { // optional::iterator and optional::const_iterator satisfy the Cpp17RandomAccessIterator and contiguous iterator.
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    static_assert(std::contiguous_iterator<decltype(it)>);
    static_assert(std::contiguous_iterator<decltype(it2)>);

    static_assert(std::random_access_iterator<decltype(it)>);
    static_assert(std::random_access_iterator<decltype(it2)>);
  }

  { // const_iterator::value_type == std::remove_cvref_t<T>, const_iterator::reference == const T&, iterator::value_type = std::remove_cvref_t<T>, iterator::reference == T&
    // std::remove_cv_t is impossible for optional<T&>
    auto it  = opt.begin();
    auto it2 = std::as_const(opt).begin();
    static_assert(std::is_same_v<typename decltype(it)::value_type, std::remove_cvref_t<T>>);
    static_assert(std::is_same_v<typename decltype(it)::reference, std::remove_reference_t<T>&>);
    static_assert(std::is_same_v<typename decltype(it2)::value_type, std::remove_cvref_t<T>>);

    // optional<T&> doesn't have const_iterator
    if constexpr (!std::is_lvalue_reference_v<T>) {
      static_assert(std::is_same_v<typename decltype(it2)::reference, const std::remove_reference_t<T>&>);
    }
  }

  { // std::ranges::size for an engaged optional<T> == 1, disengaged optional<T> == 0
    const std::optional<T> disengaged{std::nullopt};
    std::optional<T> disengaged2{std::nullopt};
    assert(std::ranges::size(opt) == 1);
    assert(std::ranges::size(std::as_const(opt)) == 1);

    assert(std::ranges::size(disengaged) == 0);
    assert(std::ranges::size(disengaged2) == 0);
  }

  { // std::ranges::enable_view<optional<T>> == true, and std::format_kind<optional<T>> == std::range_format::disabled
    static_assert(std::ranges::enable_view<std::optional<T>> == true);
    static_assert(std::format_kind<std::optional<T>> == std::range_format::disabled);
  }

  // An optional with value that is reset will have a begin() == end(), then when it is reassigned a value,
  // begin() != end(), and *begin() will contain the new value.
  {
    std::optional<T> val{v};
    assert(val.begin() != val.end());
    val.reset();
    assert(val.begin() == val.end());
    val.emplace(v);
    assert(val.begin() != val.end());
    assert(*(val.begin()) == v);
  }
}

constexpr bool test() {
  // Verify that iterator and const_iterator are present for object type T, but for T&,
  // that only iterator is available iff T is an object type and is not an unbounded array.

  static_assert(has_both_iterators<std::optional<int>>);
  static_assert(has_both_iterators<std::optional<const int>>);
  static_assert(only_has_iterator<std::optional<int&>>);
  static_assert(only_has_iterator<std::optional<const int&>>);
  static_assert(only_has_iterator<std::optional<int (&)[1]>>);
  static_assert(has_no_iterators<std::optional<int (&)[]>>);
  static_assert(has_no_iterators<std::optional<int (&)()>>);

  test<int>(1);
  test<char>('a');
  test<bool>(true);
  test<const int>(2);
  test<const char>('b');
  test<int&>(1);
  test<char&>('a');
  test<bool&>(true);
  test<const int&>(2);
  test<const char&>('b');

  static_assert(!std::ranges::range<std::optional<int (&)()>>);
  static_assert(!std::ranges::range<std::optional<int (&)[]>>);
  static_assert(std::ranges::range<std::optional<int (&)[42]>>);

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
