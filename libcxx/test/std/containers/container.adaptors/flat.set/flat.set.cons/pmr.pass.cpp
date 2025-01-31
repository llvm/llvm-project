//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: availability-pmr-missing

// <flat_set>

// Test various constructors with pmr

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_set>
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
    // flat_set(const Allocator& a);
    using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::polymorphic_allocator<int> pa = &mr;
    auto m1                                 = M(pa);
    assert(m1.empty());
    assert(std::move(m1).extract().get_allocator() == pa);
    auto m2 = M(&mr);
    assert(m2.empty());
    assert(std::move(m2).extract().get_allocator() == pa);
  }
  {
    // flat_set(const key_compare& comp, const Alloc& a);
    using M = std::flat_set<int, std::function<bool(int, int)>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    vm.emplace_back(std::greater<int>());
    assert(vm[0] == M{});
    assert(vm[0].key_comp()(2, 1) == true);
    assert(vm[0].value_comp()(2, 1) == true);
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
    //          const Allocator& a);
    using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    assert(ks.get_allocator().resource() != &mr);
    vm.emplace_back(ks);
    assert(ks.size() == 9); // ks' value is unchanged, since it was an lvalue above
    assert((vm[0] == M{1, 2, 3}));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(const flat_set&, const allocator_type&);
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({1, 2, 3}, C(5), &mr1);
    M m  = {mo, &mr2}; // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    auto keys = std::move(m).extract();
    assert((keys == std::pmr::vector<int>{1, 2, 3}));
    assert(keys.get_allocator().resource() == &mr2);

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    auto keys2 = std::move(mo).extract();
    assert((keys2 == std::pmr::vector<int>{1, 2, 3}));
    assert(keys2.get_allocator().resource() == &mr1);
  }
  {
    // flat_set(const flat_set&, const allocator_type&);
    using M = std::flat_set<int, std::less<>, std::pmr::vector<int>>;
    std::pmr::vector<M> vs;
    M m = {1, 2, 3};
    vs.push_back(m);
    assert(vs[0] == m);
  }
  {
    // flat_set& operator=(const flat_set& m);
    // pmr allocator is not propagated
    using M = std::flat_set<int, std::less<>, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({1, 2, 3}, &mr1);
    M m  = M({4, 5}, &mr2);
    m    = mo;
    assert((m == M{1, 2, 3}));
    assert(std::move(m).extract().get_allocator().resource() == &mr2);

    // mo is unchanged
    assert((mo == M{1, 2, 3}));
    assert(std::move(mo).extract().get_allocator().resource() == &mr1);
  }
  {
    // flat_set(const flat_set& m);
    using C = test_less<int>;
    std::pmr::monotonic_buffer_resource mr;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    auto mo = M({1, 2, 3}, C(5), &mr);
    auto m  = mo;

    assert(m.key_comp() == C(5));
    assert((m == M{1, 2, 3}));
    auto ks = std::move(m).extract();
    assert(ks.get_allocator().resource() == std::pmr::get_default_resource());

    // mo is unchanged
    assert(mo.key_comp() == C(5));
    assert((mo == M{1, 2, 3}));
    auto kso = std::move(mo).extract();
    assert(kso.get_allocator().resource() == &mr);
  }
  {
    //  flat_set(initializer_list<value_type> il, const Alloc& a);
    using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {3, 1, 4, 1, 5};
    vm.emplace_back(il);
    assert((vm[0] == M{1, 3, 4, 5}));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    //  flat_set(initializer_list<value_type> il, const key_compare& comp, const Alloc& a);
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::deque<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::initializer_list<M::value_type> il = {3, 1, 4, 1, 5};
    vm.emplace_back(il, C(5));
    assert((vm[0] == M{1, 3, 4, 5}));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
    assert(vm[0].key_comp() == C(5));
  }
  {
    // flat_set(InputIterator first, InputIterator last, const Allocator& a);
    int ar[]       = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    int expected[] = {1, 2, 3};
    {
      //  cpp17 iterator
      using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 9));
      assert(std::ranges::equal(vm[0], expected));
      assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(ar, ar);
      assert(vm[0].empty());
      assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
    }
  }
  {
    // flat_set(flat_set&&, const allocator_type&);
    int expected[] = {1, 2, 3};
    using C        = test_less<int>;
    using M        = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo = M({1, 3, 1, 2}, C(5), &mr1);
    M m  = {std::move(mo), &mr2}; // also test the implicitness of this constructor

    assert(m.key_comp() == C(5));
    assert(m.size() == 3);
    assert(std::equal(m.begin(), m.end(), expected, expected + 3));
    assert(std::move(m).extract().get_allocator().resource() == &mr2);

    // The original flat_set is moved-from.
    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    assert(mo.key_comp() == C(5));
    assert(std::move(mo).extract().get_allocator().resource() == &mr1);
  }
  {
    // flat_set(flat_set&&, const allocator_type&);
    using M = std::flat_set<int, std::less<>, std::pmr::deque<int>>;
    std::pmr::vector<M> vs;
    M m = {1, 3, 1, 2};
    vs.push_back(std::move(m));
    assert((std::move(vs[0]).extract() == std::pmr::deque<int>{1, 2, 3}));
  }
  {
    // flat_set& operator=(flat_set&&);
    using M = std::flat_set<std::pmr::string, std::less<>, std::pmr::vector<std::pmr::string>>;
    std::pmr::monotonic_buffer_resource mr1;
    std::pmr::monotonic_buffer_resource mr2;
    M mo =
        M({"short", "very long string that definitely won't fit in the SSO buffer and therefore becomes empty on move"},
          &mr1);
    M m = M({"don't care"}, &mr2);
    m   = std::move(mo);
    assert(m.size() == 2);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.begin()->get_allocator().resource() == &mr2);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    mo.insert("foo");
    assert(mo.begin()->get_allocator().resource() == &mr1);
  }
  {
    //  flat_set(from_range_t, R&&, const Alloc&);
    int ar[]       = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    int expected[] = {1, 2, 3};
    {
      // input_range
      using M    = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
      using Iter = cpp20_input_iterator<const int*>;
      using Sent = sentinel_wrapper<Iter>;
      using R    = std::ranges::subrange<Iter, Sent>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(std::from_range, R(Iter(ar), Sent(Iter(ar + 9))));
      assert(std::ranges::equal(vm[0], expected));
      assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
      using R = std::ranges::subrange<const int*>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      vm.emplace_back(std::from_range, R(ar, ar));
      assert(vm[0].empty());
      assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
    }
  }
  {
    // flat_set(sorted_unique_t, const container_type& key_cont, const Alloc& a);
    using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 2, 4, 10};
    vm.emplace_back(std::sorted_unique, ks);
    assert(!ks.empty()); // it was an lvalue above
    assert((vm[0] == M{1, 2, 4, 10}));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(sorted_unique_t, const container_type& key_cont,const Alloc& a);
    using M = std::flat_set<int, std::less<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks({1, 2, 4, 10}, &mr);
    vm.emplace_back(std::sorted_unique, ks);
    assert((vm[0] == M{1, 2, 4, 10}));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(sorted_unique_t, initializer_list<value_type> il, const Alloc& a);
    // cpp_17
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    int ar[] = {1, 2, 4, 5};
    vm.emplace_back(
        std::sorted_unique, cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 4), C(3));
    assert((vm[0] == M{1, 2, 4, 5}));
    assert(vm[0].key_comp() == C(3));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(sorted_unique_t, initializer_list<value_type> il, const Alloc& a);
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    int ar[1] = {42};
    vm.emplace_back(std::sorted_unique, ar, ar, C(4));
    assert(vm[0] == M{});
    assert(vm[0].key_comp() == C(4));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(InputIterator first, InputIterator last, const Alloc& a);
    // cpp_17
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    int ar[] = {1, 2, 4, 5};
    vm.emplace_back(
        std::sorted_unique, cpp17_input_iterator<const int*>(ar), cpp17_input_iterator<const int*>(ar + 4), C(3));
    assert((vm[0] == M{1, 2, 4, 5}));
    assert(vm[0].key_comp() == C(3));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }
  {
    // flat_set(InputIterator first, InputIterator last, const Alloc& a);
    using C = test_less<int>;
    using M = std::flat_set<int, C, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    int ar[1] = {42};
    vm.emplace_back(std::sorted_unique, ar, ar, C(4));
    assert(vm[0] == M{});
    assert(vm[0].key_comp() == C(4));
    assert(std::move(vm[0]).extract().get_allocator().resource() == &mr);
  }

  return 0;
}
