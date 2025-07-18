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
// constexpr iterator optional::begin() noexcept;
// constexpr iterator optional::end() noexcept;
// constexpr const_iterator optional::begin() noexcept;
// constexpr const_iterator optional::end() noexcept;

#include <cassert>
#include <concepts>
#include <iterator>
#include <optional>
#include <ranges>
#include <type_traits>

#include "test_macros.h"
#include "check_assertion.h"

constexpr bool test_concepts() {
  const std::optional<char> opt{'a'};
  std::optional<char> nonconst_opt{'n'};

  assert(std::contiguous_iterator<decltype(nonconst_opt.begin())>);
  assert(std::contiguous_iterator<decltype(nonconst_opt.end())>);
  assert(std::random_access_iterator<decltype(nonconst_opt.begin())>);
  assert(std::random_access_iterator<decltype(nonconst_opt.end())>);

  assert(std::contiguous_iterator<decltype(opt.begin())>);
  assert(std::contiguous_iterator<decltype(opt.end())>);
  assert(std::random_access_iterator<decltype(opt.begin())>);
  assert(std::random_access_iterator<decltype(opt.end())>);

  return true;
}

constexpr bool test_types() {
  const std::optional<char> opt{'a'};
  std::optional<char> nonconst_opt{'n'};
  
  assert((std::is_same_v<typename decltype(opt.begin())::value_type, char>));
  assert((std::is_same_v<typename decltype(opt.begin())::reference, const char&>));
  assert((std::is_same_v<typename decltype(nonconst_opt.begin())::value_type, char>));
  assert((std::is_same_v<typename decltype(nonconst_opt.begin())::reference, char&>));
  return true; 
}

constexpr bool test_size() {
  const std::optional<char> opt{'a'};
  std::optional<char> unengaged_opt;

  assert(std::ranges::size(opt) == 1);
  assert(std::ranges::size(unengaged_opt) == 0);

  return true;
}

constexpr bool test_begin_end() {
  const std::optional<char> opt{'a'};
  std::optional<char> unengaged_opt;

  assert(opt.begin() != opt.end());
  assert(unengaged_opt.begin() == unengaged_opt.end());
  return true;
}

constexpr bool test_noexcept() {
  const std::optional<char> opt{'a'};
  std::optional<char> nonconst_opt{'n'};

  ASSERT_NOEXCEPT(opt.begin());
  ASSERT_NOEXCEPT(opt.end());
  ASSERT_NOEXCEPT(nonconst_opt.begin());
  ASSERT_NOEXCEPT(nonconst_opt.end());

  return true;
}

constexpr bool test_value() {
  const std::optional<char> opt{'a'};
  std::optional<char> nonconst_opt{'n'};

  assert(*(opt.begin()) == 'a');
  assert(*(nonconst_opt.begin()) == 'n');
  return true;
}

constexpr bool test_syn() {
  assert(std::ranges::enable_view<std::optional<char>> == true);
  assert(std::format_kind<std::optional<char>> == std::range_format::disabled);
  return true;
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
  // 1: optional::iterator and optional const_iterator satisfy contiguous_iterator and random_access_iterator
  {
    test_concepts();
    static_assert(test_concepts());
  }

  // 2: The value types and reference types for optional::iterator and optional::const_iterator are {_Tp, _Tp&} and {_Tp, const _Tp&} respectively
  {
    test_types();
    static_assert(test_types());
  }

  // 3: The optional::begin() and optional::end() are marked noexcept.
  {
    test_noexcept();
    static_assert(test_noexcept());
  }

  // 4: optionals that have a value have begin() != end(), whereas one that doesn't has begin() == end();
  {
    test_begin_end();
    static_assert(test_begin_end());
  }

  // 5: The corresponding size for the following optionals is respected: has_value() == 1, !has_value() == 0
  {
    test_size();
    static_assert(test_size());
  }

  // 6: Dereferencing an engaged optional's iterator returns the correct value.
  {
    test_value();
    static_assert(test_value());
  }

  // 7: std::ranges::enable_view<optional<T>> == true, and std::format_kind<optional<T>> == true
  {
    test_syn();
    static_assert(test_syn());
  }

  // 8: An optional with value that is reset will have a begin() == end(), then when it is reassigned a value, begin() != end(), and *begin() will contain the new value.
  {
    test_reset();
    static_assert(test_reset());
  }

  return 0;
}
