//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr ~expected();
//
// Effects: If has_value() is true, destroys val, otherwise destroys unex.
//
// Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>
#include <memory>

#include "test_macros.h"

// Test Remarks: If is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true,
// then this destructor is a trivial destructor.
struct NonTrivial {
  ~NonTrivial() {}
};

static_assert(std::is_trivially_destructible_v<std::expected<int, int>>);
static_assert(!std::is_trivially_destructible_v<std::expected<NonTrivial, int>>);
static_assert(!std::is_trivially_destructible_v<std::expected<int, NonTrivial>>);
static_assert(!std::is_trivially_destructible_v<std::expected<NonTrivial, NonTrivial>>);

struct TrackedDestroy {
  bool& destroyed;
  constexpr TrackedDestroy(bool& b) : destroyed(b) {}
  constexpr ~TrackedDestroy() { destroyed = true; }
};

constexpr bool test() {
  // has value
  {
    bool valueDestroyed = false;
    { [[maybe_unused]] std::expected<TrackedDestroy, TrackedDestroy> e(std::in_place, valueDestroyed); }
    assert(valueDestroyed);
  }

  // has error
  {
    bool errorDestroyed = false;
    { [[maybe_unused]] std::expected<TrackedDestroy, TrackedDestroy> e(std::unexpect, errorDestroyed); }
    assert(errorDestroyed);
  }

  return true;
}

int main(int, char**) {
  std::expected<std::unique_ptr<int>, int> a = std::make_unique<int>(42);

  test();
  static_assert(test());
  return 0;
}
