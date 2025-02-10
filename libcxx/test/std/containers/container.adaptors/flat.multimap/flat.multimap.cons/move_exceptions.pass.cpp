//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// <flat_map>

// flat_multimap(flat_multimap&& s);
// If any member function in [flat.multimap.defn] exits via an exception, the invariant is restored.

#include <algorithm>
#include <cassert>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"

static int countdown = 0;

struct EvilContainer : std::vector<int> {
  EvilContainer() = default;
  EvilContainer(EvilContainer&& rhs) {
    // Throw on move-construction.
    if (--countdown == 0) {
      rhs.insert(rhs.end(), 0);
      rhs.insert(rhs.end(), 0);
      throw 42;
    }
  }
};

int main(int, char**) {
  {
    using M   = std::flat_multimap<int, int, std::less<int>, EvilContainer, std::vector<int>>;
    M mo      = {{1, 1}, {1, 2}, {3, 3}};
    countdown = 1;
    try {
      M m = std::move(mo);
      assert(false); // not reached
    } catch (int x) {
      assert(x == 42);
    }
    // The source flat_multimap maintains its class invariant.
    check_invariant(mo);
    LIBCPP_ASSERT(mo.empty());
  }
  {
    using M   = std::flat_multimap<int, int, std::less<int>, std::vector<int>, EvilContainer>;
    M mo      = {{1, 1}, {1, 2}, {3, 3}};
    countdown = 1;
    try {
      M m = std::move(mo);
      assert(false); // not reached
    } catch (int x) {
      assert(x == 42);
    }
    // The source flat_multimap maintains its class invariant.
    check_invariant(mo);
    LIBCPP_ASSERT(mo.empty());
  }
  return 0;
}
