//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// constexpr iterator optional::end() noexcept;
// constexpr const_iterator optional::end() const noexcept;

#include <cassert>
#include <iterator>
#include <optional>
#include <ranges>
#include <utility>

template <typename T>
constexpr bool test() {
  std::optional<T> disengaged{std::nullopt};

  { // end() is marked noexcept
    static_assert(noexcept(disengaged.end()));
    static_assert(noexcept(std::as_const(disengaged).end()));
  }

  { // end() == begin() and end() == end() if the optional is disengaged
    auto it  = disengaged.end();
    auto it2 = std::as_const(disengaged).end();

    assert(it == disengaged.begin());
    assert(disengaged.begin() == it);
    assert(it == disengaged.end());

    assert(it2 == std::as_const(disengaged).begin());
    assert(std::as_const(disengaged).begin() == it2);
    assert(it2 == std::as_const(disengaged).end());
  }

  std::optional<T> engaged{T{}};

  { // end() != begin() if the optional is engaged
    auto it  = engaged.end();
    auto it2 = std::as_const(engaged).end();

    assert(it != engaged.begin());
    assert(engaged.begin() != it);

    assert(it2 != std::as_const(engaged).begin());
    assert(std::as_const(engaged).begin() != it2);
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
