//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(initializer_list<value_type> il, const key_compare& comp = key_compare());
// template<class Alloc> flat_map(initializer_list<value_type> il, const Alloc& a);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "min_allocator.h"

struct DefaultCtableComp {
  explicit DefaultCtableComp() { default_constructed_ = true; }
  bool operator()(int, int) const { return false; }
  bool default_constructed_ = false;
};

int main(int, char**)
{
  std::pair<int, short> expected[] = {{1,1}, {2,2}, {3,3}, {5,2}};
  {
    using M = std::flat_map<int, short>;
    M m = {{5,2}, {2,2}, {2,2}, {3,3}, {1,1}, {3,3}};
    assert(std::equal(m.begin(), m.end(), expected, expected+4));
  }
  {
    using M = std::flat_map<int, short, std::greater<int>, std::deque<int, min_allocator<int>>>;
    M m = {{5,2}, {2,2}, {2,2}, {3,3}, {1,1}, {3,3}};
    assert(std::equal(m.rbegin(), m.rend(), expected, expected+4));
  }
  {
    using M = std::flat_map<int, short>;
    std::initializer_list<M::value_type> il = {{5,2}, {2,2}, {2,2}, {3,3}, {1,1}, {3,3}};
    M m = il;
    assert(std::equal(m.begin(), m.end(), expected, expected+4));
    static_assert( std::is_constructible_v<M, std::initializer_list<std::pair<int, short>>>);
    static_assert( std::is_constructible_v<M, std::initializer_list<std::pair<int, short>>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, short>>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, short>>, std::allocator<int>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, const short>>>);
    static_assert(!std::is_constructible_v<M, std::initializer_list<std::pair<const int, const short>>, std::allocator<int>>);
  }
  {
    using A = explicit_allocator<int>;
    {
      using M = std::flat_map<int, int, DefaultCtableComp, std::vector<int, A>, std::deque<int, A>>;
      M m = {{1,1}, {2,2}, {3,3}};
      assert(m.size() == 1);
      assert(m.begin()->first == m.begin()->second);
      LIBCPP_ASSERT(*m.begin() == std::make_pair(1, 1));
      assert(m.key_comp().default_constructed_);
    }
    {
      using M = std::flat_map<int, int, std::greater<int>, std::deque<int, A>, std::vector<int, A>>;
      A a;
      M m({{5,2}, {2,2}, {2,2}, {3,3}, {1,1}, {3,3}}, a);
      assert(std::equal(m.rbegin(), m.rend(), expected, expected+4));
    }
  }
  {
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {{3,3}, {1,1}, {4,4}, {1,1}, {5,5}};
    vm.emplace_back(il);
    assert((vm[0] == M{{1,1}, {3,3}, {4,4}, {5,5}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  return 0;
}
