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
//   flat_map(sorted_unique_t, InputIterator first, InputIterator last, const key_compare& comp = key_compare());
// template<class InputIterator, class Allocator>
//   flat_map(sorted_unique_t, InputIterator first, InputIterator last, const key_compare& comp, const Allocator& a);

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
  {
    using M = std::flat_map<int, int, std::function<bool(int, int)>>;
    using P = std::pair<int, int>;
    P ar[]  = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    auto m  = M(std::sorted_unique,
               cpp17_input_iterator<const P*>(ar),
               cpp17_input_iterator<const P*>(ar + 4),
               std::less<int>());
    assert(m == M({{1, 1}, {2, 2}, {4, 4}, {5, 5}}, std::less<>()));
    assert(m.key_comp()(1, 2) == true);
  }
  {
    using M = std::flat_map<int, int, std::greater<int>, std::deque<int, min_allocator<int>>, std::vector<int>>;
    using P = std::pair<int, int>;
    P ar[]  = {{5, 5}, {4, 4}, {2, 2}, {1, 1}};
    auto m  = M(std::sorted_unique,
               cpp17_input_iterator<const P*>(ar),
               cpp17_input_iterator<const P*>(ar + 4),
               std::greater<int>());
    assert((m == M{{5, 5}, {4, 4}, {2, 2}, {1, 1}}));
  }
  {
    // Test when the operands are of array type (also contiguous iterator type)
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>;
    std::pair<int, int> ar[1] = {{42, 42}};
    auto m                    = M(std::sorted_unique, ar, ar, C(5));
    assert(m.empty());
    assert(m.key_comp() == C(5));
  }
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<short>;
    using M  = std::flat_map<int, short, C, std::vector<int, A1>, std::deque<short, A2>>;
    using P  = std::pair<int, int>;
    P ar[]   = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    auto m   = M(std::sorted_unique, ar, ar + 4, C(3), A1(5));
    assert((m == M{{1, 1}, {2, 2}, {4, 4}, {5, 5}}));
    assert(m.key_comp() == C(3));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using A1 = test_allocator<short>;
    using A2 = test_allocator<int>;
    using M  = std::flat_map<short, int, std::less<int>, std::deque<short, A1>, std::vector<int, A2>>;
    using P  = std::pair<int, int>;
    P ar[]   = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    M m      = {std::sorted_unique, ar, ar + 4, {}, A1(5)}; // implicit ctor
    assert((m == M{{1, 1}, {2, 2}, {4, 4}, {5, 5}}));
    assert(m.keys().get_allocator() == A1(5));
    assert(m.values().get_allocator() == A2(5));
  }
  {
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    using P = std::pair<int, int>;
    P ar[]  = {{1, 1}, {2, 2}, {4, 4}, {5, 5}};
    vm.emplace_back(
        std::sorted_unique, cpp17_input_iterator<const P*>(ar), cpp17_input_iterator<const P*>(ar + 4), C(3));
    assert((vm[0] == M{{1, 1}, {2, 2}, {4, 4}, {5, 5}}));
    assert(vm[0].key_comp() == C(3));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pair<int, int> ar[1] = {{42, 42}};
    vm.emplace_back(std::sorted_unique, ar, ar, C(4));
    assert(vm[0] == M{});
    assert(vm[0].key_comp() == C(4));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  return 0;
}
