//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<class Allocator>
//   explicit flat_set(const Allocator& a);

#include <cassert>
#include <flat_set>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../test_compare.h"

int main(int, char**) {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<container_type, Alloc> is true.

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using V1 = std::vector<int, A1>;
    using V2 = std::vector<int, A2>;
    using M1 = std::flat_set<int, C, V1>;
    using M2 = std::flat_set<int, C, V2>;
    static_assert(std::is_constructible_v<M1, const A1&>);
    static_assert(std::is_constructible_v<M2, const A2&>);
    static_assert(!std::is_constructible_v<M1, const A2&>);
    static_assert(!std::is_constructible_v<M2, const A1&>);
  }
  {
    // explicit
    using M = std::flat_set<int, std::less<int>, std::vector<int, test_allocator<int>>>;

    static_assert(std::is_constructible_v<M, test_allocator<int>>);
    static_assert(!std::is_convertible_v<test_allocator<int>, M>);
  }
  {
    using A = test_allocator<short>;
    using M = std::flat_set<int, std::less<int>, std::vector<int, test_allocator<int>>>;
    M m(A(0, 5));
    assert(m.empty());
    assert(m.begin() == m.end());
    auto v = std::move(m).extract();
    assert(v.get_allocator().get_id() == 5);
  }

  return 0;
}
