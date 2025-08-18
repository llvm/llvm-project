//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// constexpr iterator optional::begin() noexcept;
// constexpr const_iterator optional::begin() const noexcept;

#include <cassert>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

template <typename T>
constexpr bool test() {
  std::optional<T> opt{T{}};

  { // begin() is marked noexcept
    static_assert(noexcept(opt.begin()));
    static_assert(noexcept(std::as_const(opt).begin()));
  }

  { // Dereferencing an iterator at the beginning == indexing the 0th element, and that calling begin() again return the same iterator.
    auto iter1 = opt.begin();
    auto iter2 = std::as_const(opt).begin();
    assert(*iter1 == iter1[0]);
    assert(*iter2 == iter2[0]);
    assert(iter1 == opt.begin());
    assert(iter2 == std::as_const(opt).begin());
  }

  { // Calling begin() multiple times on a disengaged optional returns the same iterator.
    std::optional<T> disengaged{std::nullopt};
    auto iter1 = disengaged.begin();
    auto iter2 = std::as_const(disengaged).begin();
    assert(iter1 == disengaged.begin());
    assert(iter2 == std::as_const(disengaged).begin());
  }

  return true;
}

constexpr bool tests() {
  assert(test<int>());
  assert(test<char>());
  assert(test<const int>());
  assert(test<const char>());
  return true;
}

int main(int, char**) {
  assert(tests());
  static_assert(tests());

  return 0;
}
