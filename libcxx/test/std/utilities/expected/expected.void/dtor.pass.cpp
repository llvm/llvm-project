//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Older Clangs do not support the C++20 feature to constrain destructors
// XFAIL: apple-clang-14

// constexpr ~expected();
//
// Effects: If has_value() is false, destroys unex.
//
// Remarks: If is_trivially_destructible_v<E> is true, then this destructor is a trivial destructor.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test Remarks: If is_trivially_destructible_v<E> is true, then this destructor is a trivial destructor.
struct NonTrivial {
  ~NonTrivial() {}
};

static_assert(std::is_trivially_destructible_v<std::expected<void, int>>);
static_assert(!std::is_trivially_destructible_v<std::expected<void, NonTrivial>>);

struct TrackedDestroy {
  bool& destroyed;
  constexpr TrackedDestroy(bool& b) : destroyed(b) {}
  constexpr ~TrackedDestroy() { destroyed = true; }
};

constexpr bool test() {
  // has value
  { [[maybe_unused]] std::expected<void, TrackedDestroy> e(std::in_place); }

  // has error
  {
    bool errorDestroyed = false;
    { [[maybe_unused]] std::expected<void, TrackedDestroy> e(std::unexpect, errorDestroyed); }
    assert(errorDestroyed);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
