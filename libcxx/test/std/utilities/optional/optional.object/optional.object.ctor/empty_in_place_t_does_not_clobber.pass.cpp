//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr optional(in_place_t);

// Test that the SFINAE "is_constructible<value_type>" isn't evaluated by the
// in_place_t constructor with no arguments when the Clang is trying to check
// copy constructor.

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "archetypes.hpp"

using std::optional;

struct Wrapped {
  struct Inner {
    bool Dummy = true;
  };
  std::optional<Inner> inner;
};

int main(int, char**) {
  static_assert(std::is_default_constructible<Wrapped::Inner>::value, "");
  Wrapped w;
  w.inner.emplace();
  assert(w.inner.has_value());

  return 0;
}
