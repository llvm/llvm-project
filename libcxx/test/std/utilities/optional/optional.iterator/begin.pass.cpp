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
// constexpr const_iterator optional::begin() noexcept;

#include <cassert>
#include <iterator>
#include <optional>
#include <type_traits>

template <typename T>
constexpr bool test() {
  constexpr std::optional<T> opt{T{}};
  std::optional<T> nonconst_opt{T{}};

  { // begin() is marked noexcept
    assert(noexcept(opt.begin()));
    assert(noexcept(nonconst_opt.begin()));
  }

  { // Dereferencing an iterator at the beginning == indexing the 0th element, and that calling begin() again return the same iterator.
    auto iter1 = opt.begin();
    auto iter2 = nonconst_opt.begin();
    assert(*iter1 == iter1[0]);
    assert(*iter2 == iter2[0]);
    assert(iter1 == opt.begin());
    assert(iter2 == nonconst_opt.begin());
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

  return 0;
}
