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
// constexpr const_iterator optional::end() noexcept;

#include <cassert>
#include <iterator>
#include <ranges>
#include <optional>

template <typename T>
constexpr bool test() {
  std::optional<T> unengaged{std::nullopt};
  constexpr std::optional<T> unengaged2{std::nullopt};

  { // end() is marked noexcept
    assert(noexcept(unengaged.end()));
    assert(noexcept(unengaged2.end()));
  }

  { // end() == begin() and end() == end() if the optional is unengaged
    auto it  = unengaged.end();
    auto it2 = unengaged2.end();

    assert(it == unengaged.begin());
    assert(unengaged.begin() == it);
    assert(it == unengaged.end());

    assert(it2 == unengaged2.begin());
    assert(unengaged2.begin() == it2);
    assert(it2 == unengaged2.end());
  }

  std::optional<T> engaged{T{}};
  constexpr std::optional<T> engaged2{T{}};

  { // end() != begin() if the optional is engaged
    auto it  = engaged.end();
    auto it2 = engaged2.end();

    assert(it != engaged.begin());
    assert(engaged.begin() != it);

    assert(it2 != engaged2.begin());
    assert(engaged2.begin() != it2);
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

int main() {
  assert(tests());
  static_assert(tests());
}
