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
//   flat_map(InputIterator first, InputIterator last, const key_compare& comp, const Allocator& a);

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
#include "../../../test_compare.h"

int main(int, char**) {
  using P      = std::pair<int, short>;
  P ar[]       = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {3, 9}};
  P expected[] = {{1, 1}, {2, 4}, {3, 6}};
  {
    using M = std::flat_map<int, short, std::function<bool(int, int)>>;
    auto m  = M(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9), std::less<int>());
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.key_comp()(1, 2) == true);
  }
  {
    using M = std::flat_map<int, short, std::greater<int>, std::deque<int, min_allocator<int>>>;
    auto m  = M(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9), std::greater<int>());
    assert(std::ranges::equal(m.keys(), expected | std::views::reverse | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected | std::views::reverse));
  }
  {
    // Test when the operands are of array type (also contiguous iterator type)
    using C = test_less<int>;
    using M = std::flat_map<int, short, C, std::vector<int, min_allocator<int>>>;
    auto m  = M(ar, ar, C(5));
    assert(m.empty());
    assert(m.key_comp() == C(5));
  }
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, C, std::vector<int, A1>, std::deque<short, A2>>;
    auto m   = M(ar, ar + 9, C(3), A1(5));
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.key_comp() == C(3));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, std::less<int>, std::deque<int, A1>, std::vector<short, A2>>;
    M m      = {ar, ar + 9, {}, A2(5)}; // implicit ctor
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using C = test_less<int>;
    using M = std::flat_map<int, short, C, std::pmr::vector<int>, std::pmr::deque<short>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 9), C(3));
    assert(std::ranges::equal(vm[0].keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(vm[0], expected));
    assert(vm[0].key_comp() == C(3));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    using C = test_less<int>;
    using M = std::flat_map<int, short, C, std::pmr::vector<int>, std::pmr::vector<short>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(ar, ar, C(4));
    assert(vm[0].empty());
    assert(vm[0].key_comp() == C(4));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  return 0;
}
