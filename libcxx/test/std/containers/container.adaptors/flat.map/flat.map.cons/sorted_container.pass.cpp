//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(sorted_unique_t, key_container_type key_cont, mapped_container_type mapped_cont,
//          const key_compare& comp = key_compare());

#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <vector>

#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

int main(int, char**) {
  {
    using M              = std::flat_map<int, char>;
    std::vector<int> ks  = {1, 2, 4, 10};
    std::vector<char> vs = {4, 3, 2, 1};
    auto m               = M(std::sorted_unique, ks, vs);
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    m = M(std::sorted_unique, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
  }
  {
    using Ks = std::deque<int, min_allocator<int>>;
    using Vs = std::deque<char, min_allocator<char>>;
    using M  = std::flat_map<int, char, std::greater<int>, Ks, Vs>;
    Ks ks    = {10, 4, 2, 1};
    Vs vs    = {1, 2, 3, 4};
    auto m   = M(std::sorted_unique, ks, vs);
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    m = M(std::sorted_unique, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
  }
  {
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 2, 4, 10}, A(4));
    auto vs = std::deque<int, A>({4, 3, 2, 1}, A(5));
    auto m  = M(std::sorted_unique, std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(5));
  }
  {
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 2, 4, 10}, A(4));
    auto vs = std::deque<int, A>({4, 3, 2, 1}, A(5));
    auto m  = M(std::sorted_unique, ks, vs, A(6)); // replaces the allocators
    assert(!ks.empty());                           // it was an lvalue above
    assert(!vs.empty());                           // it was an lvalue above
    assert((m == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    assert(m.keys().get_allocator() == A(6));
    assert(m.values().get_allocator() == A(6));
  }
  {
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 2, 4, 10};
    std::pmr::vector<int> vs = {4, 3, 2, 1};
    vm.emplace_back(std::sorted_unique, ks, vs);
    assert(!ks.empty()); // it was an lvalue above
    assert(!vs.empty()); // it was an lvalue above
    assert((vm[0] == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 2, 4, 10};
    std::pmr::vector<int> vs = {4, 3, 2, 1};
    vm.emplace_back(std::sorted_unique, std::move(ks), std::move(vs));
    LIBCPP_ASSERT(ks.size() == 4); // ks' size is unchanged, since it uses a different allocator
    LIBCPP_ASSERT(vs.size() == 4); // vs' size is unchanged, since it uses a different allocator
    assert((vm[0] == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
#if 0
  {
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks({1,2,4,10}, &mr);
    std::pmr::vector<int> vs({4,3,2,1}, &mr);
    vm.emplace_back(std::sorted_unique, std::move(ks), std::move(vs));
    assert(ks.empty()); // ks is moved-from (after LWG 3802)
    assert(vs.empty()); // vs is moved-from (after LWG 3802)
    assert((vm[0] == M{{1,4}, {2,3}, {4,2}, {10,1}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    using M = std::flat_map<MoveOnly, MoveOnly, std::less<>, std::pmr::vector<MoveOnly>, std::pmr::vector<MoveOnly>>;
    std::pmr::vector<M> vm;
    std::pmr::vector<MoveOnly> ks;
    std::pmr::vector<MoveOnly> vs;
    vm.emplace_back(std::sorted_unique, std::move(ks), std::move(vs)); // this was a hard error before LWG 3802
    assert(vm.size() == 1);
    assert(vm[0].empty());
  }
#endif
  return 0;
}
