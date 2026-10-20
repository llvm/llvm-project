//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr void emplace() noexcept;
//
// Effects: If has_value() is false, destroys unex and sets has_val to true.

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"
#include "test_macros.h"

template <class T>
concept EmplaceNoexcept =
    requires(T t) {
      { t.emplace() } noexcept;
    };
static_assert(!EmplaceNoexcept<int>);

static_assert(EmplaceNoexcept<std::expected<void, int>>);

constexpr bool test() {
  // has_value
  {
    std::expected<void, int> e;
    e.emplace();
    assert(e.has_value());
  }

  // !has_value
  {
    Traced::state state{};
    std::expected<int, Traced> e(std::unexpect, state, 5);
    e.emplace();

    assert(state.dtorCalled);
    assert(e.has_value());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
