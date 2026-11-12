//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// bool <copyable-box>::__has_value() const

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility> // in_place_t

#include "types.h"

template <class T>
constexpr void check() {
  std::ranges::__movable_box<T> const x(std::in_place, 10);
  assert(x.__has_value());
}

constexpr bool test() {
  check<CopyConstructible>();        // primary template
  check<Copyable>();                 // optimization #1
  check<NothrowCopyConstructible>(); // optimization #2
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  // Tests for the empty state. Those can't be constexpr, since they are only reached
  // through throwing an exception.
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  {
    std::ranges::__movable_box<ThrowsOnCopy> x = create_empty_box();
    assert(!x.__has_value());
  }
#endif

  return 0;
}
