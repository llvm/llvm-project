//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class Allocator>
//   explicit flat_multimap(const Allocator& a);

#include <cassert>
#include <flat_map>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../test_compare.h"

int main(int, char**) {
  {
    // The constructors in this subclause shall not participate in overload
    // resolution unless uses_allocator_v<key_container_type, Alloc> is true
    // and uses_allocator_v<mapped_container_type, Alloc> is true.

    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = other_allocator<int>;
    using V1 = std::vector<int, A1>;
    using V2 = std::vector<int, A2>;
    using M1 = std::flat_multimap<int, int, C, V1, V1>;
    using M2 = std::flat_multimap<int, int, C, V1, V2>;
    using M3 = std::flat_multimap<int, int, C, V2, V1>;
    static_assert(std::is_constructible_v<M1, const A1&>);
    static_assert(!std::is_constructible_v<M1, const A2&>);
    static_assert(!std::is_constructible_v<M2, const A2&>);
    static_assert(!std::is_constructible_v<M3, const A2&>);
  }
  {
    // explicit
    using M =
        std::flat_multimap<int,
                           long,
                           std::less<int>,
                           std::vector<int, test_allocator<int>>,
                           std::vector<long, test_allocator<long>>>;

    static_assert(std::is_constructible_v<M, test_allocator<int>>);
    static_assert(!std::is_convertible_v<test_allocator<int>, M>);
  }
  {
    using A = test_allocator<short>;
    using M =
        std::flat_multimap<int,
                           long,
                           std::less<int>,
                           std::vector<int, test_allocator<int>>,
                           std::vector<long, test_allocator<long>>>;
    M m(A(0, 5));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.keys().get_allocator().get_id() == 5);
    assert(m.values().get_allocator().get_id() == 5);
  }

  return 0;
}
