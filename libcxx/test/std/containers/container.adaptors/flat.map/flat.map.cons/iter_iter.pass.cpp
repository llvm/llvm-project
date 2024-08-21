//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template <class InputIterator>
//   flat_map(InputIterator first, InputIterator last, const key_compare& comp = key_compare());
// template<class InputIterator, class Allocator>
//   flat_map(InputIterator first, InputIterator last, const Allocator& a);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

int main(int, char**) {
  using P      = std::pair<int, short>;
  P ar[]       = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {3, 9}};
  P expected[] = {{1, 1}, {2, 4}, {3, 6}};
  {
    using M = std::flat_map<int, short>;
    auto m  = M(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    using M = std::flat_map<int, short, std::greater<int>, std::deque<int, min_allocator<int>>, std::deque<short>>;
    auto m  = M(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9));
    assert((m.keys() == std::deque<int, min_allocator<int>>{3, 2, 1}));
    LIBCPP_ASSERT((m.values() == std::deque<short>{6, 4, 1}));
  }
  {
    // Test when the operands are of array type (also contiguous iterator type)
    using M = std::flat_map<int, short, std::greater<int>, std::vector<int, min_allocator<int>>>;
    auto m  = M(ar, ar);
    assert(m.empty());
  }
  {
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::vector<int, A1>, std::deque<short, A2>>;
    auto m   = M(ar, ar + 9, A1(5));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::vector<int, A1>, std::deque<short, A2>>;
    M m      = {ar, ar + 9, A1(5)}; // implicit ctor
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using M = std::flat_map<int, short, std::less<int>, std::pmr::vector<int>, std::pmr::vector<short>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9));
    assert(std::ranges::equal(vm[0].keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(vm[0], expected));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    using M = std::flat_map<int, short, std::less<int>, std::pmr::vector<int>, std::pmr::vector<short>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(ar, ar);
    assert(vm[0].empty());
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  return 0;
}
