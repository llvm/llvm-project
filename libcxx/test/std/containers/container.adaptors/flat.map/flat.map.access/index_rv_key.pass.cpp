//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// mapped_type& operator[](key_type&& k);

#include <flat_map>
#include <functional>
#include <cassert>

#include "../helpers.h"
#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

// Constraints: is_constructible_v<mapped_type> is true.
template <class M, class Input>
concept CanIndex = requires(M m, Input k) { m[k]; };

static_assert(CanIndex<std::flat_map<int, double>, int&&>);
static_assert(!CanIndex<std::flat_map<int, NoDefaultCtr>, int&&>);

int main(int, char**) {
  {
    std::flat_map<MoveOnly, double> m;
    ASSERT_SAME_TYPE(decltype(m[MoveOnly{}]), double&);
    assert(m.size() == 0);
    assert(m[1] == 0.0);
    assert(m.size() == 1);
    m[1] = -1.5;
    assert(m[1] == -1.5);
    assert(m.size() == 1);
    assert(m[6] == 0);
    assert(m.size() == 2);
    m[6] = 6.5;
    assert(m[6] == 6.5);
    assert(m.size() == 2);
  }
  {
    std::flat_map< MoveOnly,
                   double,
                   std::less<>,
                   std::vector<MoveOnly, min_allocator<MoveOnly>>,
                   std::vector<double, min_allocator<double>>>
        m;
    ASSERT_SAME_TYPE(decltype(m[MoveOnly{}]), double&);
    assert(m.size() == 0);
    assert(m[1] == 0.0);
    assert(m.size() == 1);
    m[1] = -1.5;
    assert(m[1] == -1.5);
    assert(m.size() == 1);
    assert(m[6] == 0);
    assert(m.size() == 2);
    m[6] = 6.5;
    assert(m[6] == 6.5);
    assert(m.size() == 2);
  }

  return 0;
}
