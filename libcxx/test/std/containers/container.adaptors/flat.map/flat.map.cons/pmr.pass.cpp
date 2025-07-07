//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: availability-pmr-missing

// <flat_map>

// Test various constructors with pmr

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <ranges>
#include <vector>
#include <string>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_allocator.h"
#include "../../../test_compare.h"

int main(int, char**) {
  {
    // flat_map(const Allocator& a);
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
  {
    // flat_map(const key_compare& comp, const Alloc& a);
    using M = std::flat_map<int, int, std::function<bool(int, int)>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(std::greater<int>());
    assert(vm[0] == M{});
    assert(vm[0].key_comp()(2, 1) == true);
    assert(vm[0].value_comp()({2, 0}, {1, 0}) == true);
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    // flat_map(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
    //          const Allocator& a);
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::pmr::vector<int> vs = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    assert(ks.get_allocator().resource() != &mr);
    assert(vs.get_allocator().resource() != &mr);
    vm.emplace_back(ks, vs);
    assert(ks.size() == 9); // ks' value is unchanged, since it was an lvalue above
    assert(vs.size() == 9); // vs' value is unchanged, since it was an lvalue above
    assert((vm[0] == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    // flat_map(const flat_map&, const allocator_type&);
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({1, 2, 3}, {2, 2, 1}, C(5), &mr1);
    M m  = {mo, &mr2}; // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    assert((m.keys() == std::pmr::vector<int>{1, 2, 3}));
    assert((m.values() == std::pmr::vector<int>{2, 2, 1}));
    assert(m.keys().get_allocator().resource() == &mr2);
    assert(m.values().get_allocator().resource() == &mr2);

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert((mo.keys() == std::pmr::vector<int>{1, 2, 3}));
    assert((mo.values() == std::pmr::vector<int>{2, 2, 1}));
    assert(mo.keys().get_allocator().resource() == &mr1);
    assert(mo.values().get_allocator().resource() == &mr1);
  }
  {
    // flat_map(const flat_map&, const allocator_type&);
    using M = std::flat_map<int, int, std::less<>, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::vector<M> vs;
    M m = {{1, 2}, {2, 2}, {3, 1}};
    vs.push_back(m);
    assert(vs[0] == m);
  }
  {
    // flat_map& operator=(const flat_map& m);
    // pmr allocator is not propagated
    using M = std::flat_map<int, int, std::less<>, std::pmr::deque<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({{1, 1}, {2, 2}, {3, 3}}, &mr1);
    M m  = M({{4, 4}, {5, 5}}, &mr2);
    m    = mo;
    assert((m == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(m.keys().get_allocator().resource() == &mr2);
    assert(m.values().get_allocator().resource() == &mr2);

    // mo is unchanged
    assert((mo == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(mo.keys().get_allocator().resource() == &mr1);
  }
  {
    // flat_map(const flat_map& m);
    using C = test_less<int>;
    std::pmr::monotonic_buffer_resource mr;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::vector<int>>;
    auto mo = M({{1, 1}, {2, 2}, {3, 3}}, C(5), &mr);
    auto m  = mo;

    assert(m.key_comp() == C(5));
    assert((m == M{{1, 1}, {2, 2}, {3, 3}}));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator().resource() == std::pmr::get_default_resource());
    assert(vs.get_allocator().resource() == std::pmr::get_default_resource());

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert((mo == M{{1, 1}, {2, 2}, {3, 3}}));
    auto [kso, vso] = std::move(mo).extract();
    assert(kso.get_allocator().resource() == &mr);
    assert(vso.get_allocator().resource() == &mr);
  }
  {
    //  flat_map(initializer_list<value_type> il, const Alloc& a);
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {{3, 3}, {1, 1}, {4, 4}, {1, 1}, {5, 5}};
    vm.emplace_back(il);
    assert((vm[0] == M{{1, 1}, {3, 3}, {4, 4}, {5, 5}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    //  flat_map(initializer_list<value_type> il, const key_compare& comp, const Alloc& a);
    using C = test_less<int>;
    using M = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {{3, 3}, {1, 1}, {4, 4}, {1, 1}, {5, 5}};
    vm.emplace_back(il, C(5));
    assert((vm[0] == M{{1, 1}, {3, 3}, {4, 4}, {5, 5}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
    assert(vm[0].key_comp() == C(5));
  }
  {
    // flat_map(InputIterator first, InputIterator last, const Allocator& a);
    using P      = std::pair<int, short>;
    P ar[]       = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {3, 9}};
    P expected[] = {{1, 1}, {2, 4}, {3, 6}};
    {
      //  cpp17 iterator
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
  }
  {
    // flat_map(flat_map&&, const allocator_type&);
    std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 1}};
    using C                        = test_less<int>;
    using M                        = std::flat_map<int, int, C, std::pmr::vector<int>, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({{1, 1}, {3, 1}, {1, 1}, {2, 2}}, C(5), &mr1);
    M m  = {std::move(mo), &mr2}; // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    assert(m.size() == 3);
    assert(m.keys().get_allocator().resource() == &mr2);
    assert(m.values().get_allocator().resource() == &mr2);
    assert(std::equal(m.begin(), m.end(), expected, expected + 3));

    // The original flat_map is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.key_comp() == C(5));
    assert(mo.keys().get_allocator().resource() == &mr1);
    assert(mo.values().get_allocator().resource() == &mr1);
  }
  {
    // flat_map(flat_map&&, const allocator_type&);
    using M = std::flat_map<int, int, std::less<>, std::pmr::deque<int>, std::pmr::vector<int>>;
    std::pmr::vector<M> vs;
    M m = {{1, 1}, {3, 1}, {1, 1}, {2, 2}};
    vs.push_back(std::move(m));
    assert((vs[0].keys() == std::pmr::deque<int>{1, 2, 3}));
    assert((vs[0].values() == std::pmr::vector<int>{1, 2, 1}));
  }
  {
    // flat_map& operator=(flat_map&&);
    using M =
        std::flat_map<std::pmr::string, int, std::less<>, std::pmr::vector<std::pmr::string>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({{"short", 1},
              {"very long string that definitely won't fit in the SSO buffer and therefore becomes empty on move", 2}},
             &mr1);
    M m  = M({{"don't care", 3}}, &mr2);
    m    = std::move(mo);
    assert(m.size() == 2);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.begin()->first.get_allocator().resource() == &mr2);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    mo.insert({"foo", 1});
    assert(mo.begin()->first.get_allocator().resource() == &mr1);
  }
  {
    //  flat_map(from_range_t, R&&, const Alloc&);
    using P      = std::pair<int, short>;
    P ar[]       = {{1, 1}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {2, 7}, {3, 8}, {3, 9}};
    P expected[] = {{1, 1}, {2, 4}, {3, 6}};
    {
      // input_range
      using M    = std::flat_map<int, short, std::less<int>, std::pmr::vector<int>, std::pmr::vector<short>>;
      using Iter = cpp20_input_iterator<const P*>;
      using Sent = sentinel_wrapper<Iter>;
      using R    = std::ranges::subrange<Iter, Sent>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
      assert(std::ranges::equal(vm[0].keys(), expected | std::views::elements<0>));
      LIBCPP_ASSERT(std::ranges::equal(vm[0], expected));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_map<int, short, std::less<int>, std::pmr::vector<int>, std::pmr::vector<short>>;
      using R = std::ranges::subrange<const P*>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(std::from_range, R(ar, ar));
      assert(vm[0].empty());
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
  }
  {
    // flat_map(sorted_unique_t, const key_container_type& key_cont,
    //          const mapped_container_type& mapped_cont, const Alloc& a);
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
    // flat_map(sorted_unique_t, const key_container_type& key_cont,
    //          const mapped_container_type& mapped_cont, const Alloc& a);
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks({1, 2, 4, 10}, &mr);
    std::pmr::vector<int> vs({4, 3, 2, 1}, &mr);
    vm.emplace_back(std::sorted_unique, ks, vs);
    assert((vm[0] == M{{1, 4}, {2, 3}, {4, 2}, {10, 1}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }
  {
    // flat_map(sorted_unique_t, initializer_list<value_type> il, const Alloc& a);
    // cpp_17
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
    // flat_map(sorted_unique_t, initializer_list<value_type> il, const Alloc& a);
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
  {
    // flat_map(InputIterator first, InputIterator last, const Alloc& a);
    // cpp_17
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
    // flat_map(InputIterator first, InputIterator last, const Alloc& a);
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
