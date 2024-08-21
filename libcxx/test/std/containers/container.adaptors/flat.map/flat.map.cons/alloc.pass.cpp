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
//   explicit flat_map(const Allocator& a);

#include <cassert>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    using A = test_allocator<short>;
    using M =
        std::flat_map<int,
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
  {
    using M = std::flat_map<int, short, std::less<int>, std::pmr::vector<int>, std::pmr::vector<short>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::polymorphic_allocator<int> pa = &mr;
    auto m1                                 = M(pa);
    assert(m1.empty());
    assert(m1.keys().get_allocator() == pa);
    assert(m1.values().get_allocator() == pa);
    auto m2 = M(&mr);
    assert(m2.empty());
    assert(m2.keys().get_allocator() == pa);
    assert(m2.values().get_allocator() == pa);
  }
  return 0;
}
