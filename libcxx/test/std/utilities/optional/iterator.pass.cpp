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

// 1: optional::iterator and optional const_iterator satisfy contiguous_iterator and random_access_iterator
// 2. The value types and reference types for optional::iterator and optional::const_iterator are {_Tp, _Tp&} and {_Tp, const _Tp&} respectively
// 3: The optional::begin() and optional::end() are marked noexcept.
// 4: optionals that have a value have begin() != end(), whereas one that doesn't has begin() == end();
// 5: The corresponding size for the following optionals is respected: has_value() == 1, !has_value() == 0
// 6: Dereferencing an engaged optional's iterator returns the correct value.
// 7: std::ranges::enable_view<optional<T>> == true, and std::format_kind<optional<T>> == true
// 8: Verify that an iterator for loop counts only 1 item for an engaged optional, and 0 for an unegaged one.
// 9: An optional with value that is reset will have a begin() == end(), then when it is reassigned a value, begin() != end(), and *begin() will contain the new value.

#include <cassert>
#include <optional>

#include "test_macros.h"
#include "check_assertion.h"

constexpr int test_loop(const std::optional<char> val) {
  int size = 0;
  for (auto&& v : val) {
    std::ignore = v;
    size++;
  }
  return size;
}

constexpr bool test_reset() {
  std::optional<int> val{1};
  assert(val.begin() != val.end());
  val.reset();
  assert(val.begin() == val.end());
  val = 1;
  assert(val.begin() != val.end());
  assert(*(val.begin()) == 1);
  return true;
}

int main(int, char**) {
  constexpr const std::optional<char> opt{'a'};
  constexpr std::optional<char> unengaged_opt;
  std::optional<char> nonconst_opt{'n'};

  // 1
  {
    static_assert(std::contiguous_iterator<decltype(nonconst_opt.begin())>);
    static_assert(std::contiguous_iterator<decltype(nonconst_opt.end())>);
    static_assert(std::random_access_iterator<decltype(nonconst_opt.begin())>);
    static_assert(std::random_access_iterator<decltype(nonconst_opt.end())>);

    static_assert(std::contiguous_iterator<decltype(opt.begin())>);
    static_assert(std::contiguous_iterator<decltype(opt.end())>);
    static_assert(std::random_access_iterator<decltype(opt.begin())>);
    static_assert(std::random_access_iterator<decltype(opt.end())>);
  }

  { // 2
    static_assert(std::same_as<typename decltype(opt.begin())::value_type, char>);
    static_assert(std::same_as<typename decltype(opt.begin())::reference, const char&>);
    static_assert(std::same_as<typename decltype(nonconst_opt.begin())::value_type, char>);
    static_assert(std::same_as<typename decltype(nonconst_opt.begin())::reference, char&>);
  }

  // 3
  {
    ASSERT_NOEXCEPT(opt.begin());
    ASSERT_NOEXCEPT(opt.end());
    ASSERT_NOEXCEPT(nonconst_opt.begin());
    ASSERT_NOEXCEPT(nonconst_opt.end());
  }

  { // 4
    static_assert(opt.begin() != opt.end());
    static_assert(unengaged_opt.begin() == unengaged_opt.end());
    assert(unengaged_opt.begin() == unengaged_opt.end());
    assert(opt.begin() != opt.end());
    assert(nonconst_opt.begin() != opt.end());
  }

  // 5
  {
    static_assert(std::ranges::size(opt) == 1);
    static_assert(std::ranges::size(unengaged_opt) == 0);
    assert(std::ranges::size(opt) == 1);
    assert(std::ranges::size(unengaged_opt) == 0);
  }

  { // 6
    static_assert(*opt.begin() == 'a');
    assert(*(opt.begin()) == 'a');
    assert(*(nonconst_opt.begin()) == 'n');
  }

  { // 7
    static_assert(std::ranges::enable_view<std::optional<char>> == true);
    assert(std::format_kind<std::optional<char>> == std::range_format::disabled);
  }

  { // 8
    static_assert(test_loop(opt) == 1);
    static_assert(test_loop(unengaged_opt) == 0);
    assert(test_loop(opt) == 1);
    assert(test_loop(unengaged_opt) == 0);
  }

  { // 9
    static_assert(test_reset());
    assert(test_reset());
  }

  return 0;
}
