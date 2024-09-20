//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map(key_container_type key_cont, mapped_container_type mapped_cont,
//           const key_compare& comp = key_compare());
// template<class Allocator>
//   flat_map(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
//            const Allocator& a);
// template<class Alloc>
//   flat_map(const key_container_type& key_cont, const mapped_container_type& mapped_cont,
//            const key_compare& comp, const Alloc& a);

#include <algorithm>
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
#include "../../../test_compare.h"

struct P {
  int first;
  int second;
  template <class T, class U>
  bool operator==(const std::pair<T, U>& rhs) const {
    return MoveOnly(first) == rhs.first && MoveOnly(second) == rhs.second;
  }
};

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
    using M1 = std::flat_map<int, int, C, V1, V1>;
    using M2 = std::flat_map<int, int, C, V1, V2>;
    using M3 = std::flat_map<int, int, C, V2, V1>;
    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const A2&>);

    static_assert(std::is_constructible_v<M1, const V1&, const V1&, const C&, const A1&>);
    static_assert(!std::is_constructible_v<M1, const V1&, const V1&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M2, const V1&, const V2&, const C&, const A2&>);
    static_assert(!std::is_constructible_v<M3, const V2&, const V1&, const C&, const A2&>);
  }
  {
    // flat_map(key_container_type , mapped_container_type)
    using M              = std::flat_map<int, char>;
    std::vector<int> ks  = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<char> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m               = M(ks, vs);
    assert((m.keys() == std::vector<int>{1, 2, 3}));
    LIBCPP_ASSERT((m.values() == std::vector<char>{1, 4, 6}));

    // explicit(false)
    M m2 = {ks, vs};
    assert(m2 == m);

    m = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m.keys() == std::vector<int>{1, 2, 3}));
    LIBCPP_ASSERT((m.values() == std::vector<char>{1, 4, 6}));
  }
  {
    // flat_map(key_container_type , mapped_container_type)
    // move-only
    P expected[] = {{3, 2}, {2, 1}, {1, 3}};
    using Ks     = std::deque<int, min_allocator<int>>;
    using Vs     = std::vector<MoveOnly, min_allocator<MoveOnly>>;
    using M      = std::flat_map<int, MoveOnly, std::greater<int>, Ks, Vs>;
    Ks ks        = {1, 3, 2};
    Vs vs;
    vs.push_back(3);
    vs.push_back(2);
    vs.push_back(1);
    auto m = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert(std::ranges::equal(m, expected, std::equal_to<>()));
  }
  {
    // flat_map(key_container_type , mapped_container_type)
    // container's allocators are used
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(std::move(ks), std::move(vs));
    assert(ks.empty()); // it was moved-from
    assert(vs.empty()); // it was moved-from
    assert((m == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(6));
  }
  {
    // flat_map(key_container_type , mapped_container_type, key_compare)
    using C              = test_less<int>;
    using M              = std::flat_map<int, char, C>;
    std::vector<int> ks  = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<char> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m               = M(ks, vs, C(4));
    assert((m.keys() == std::vector<int>{1, 2, 3}));
    LIBCPP_ASSERT((m.values() == std::vector<char>{1, 4, 6}));
    assert(m.key_comp() == C(4));

    // explicit(false)
    M m2 = {ks, vs, C(4)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
  }
  {
    // flat_map(key_container_type , mapped_container_type, const Allocator&)
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    auto m  = M(ks, vs, A(4)); // replaces the allocators
    assert(!ks.empty());       // it was an lvalue above
    assert(!vs.empty());       // it was an lvalue above
    assert((m == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }
  {
    // flat_map(key_container_type , mapped_container_type, const Allocator&)
    // explicit(false)
    using A = test_allocator<int>;
    using M = std::flat_map<int, int, std::less<int>, std::vector<int, A>, std::deque<int, A>>;
    auto ks = std::vector<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(5));
    auto vs = std::deque<int, A>({1, 1, 1, 2, 2, 3, 2, 3, 3}, A(6));
    M m     = {ks, vs, A(4)}; // implicit ctor
    assert(!ks.empty());      // it was an lvalue above
    assert(!vs.empty());      // it was an lvalue above
    assert((m == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(m.keys().get_allocator() == A(4));
    assert(m.values().get_allocator() == A(4));
  }
  {
    // flat_map(key_container_type , mapped_container_type, key_compare, const Allocator&)
    using C                = test_less<int>;
    using A                = test_allocator<int>;
    using M                = std::flat_map<int, int, C, std::vector<int, A>, std::vector<int, A>>;
    std::vector<int, A> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::vector<int, A> vs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto m                 = M(ks, vs, C(4), A(5));
    assert((m.keys() == std::vector<int, A>{1, 2, 3}));
    LIBCPP_ASSERT((m.values() == std::vector<int, A>{1, 4, 6}));
    assert(m.key_comp() == C(4));
    assert(m.keys().get_allocator() == A(5));
    assert(m.values().get_allocator() == A(5));

    // explicit(false)
    M m2 = {ks, vs, C(4), A(5)};
    assert(m2 == m);
    assert(m2.key_comp() == C(4));
    assert(m2.keys().get_allocator() == A(5));
    assert(m2.values().get_allocator() == A(5));
  }
  {
    // pmr
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
    // pmr move
    using M = std::flat_map<int, int, std::less<int>, std::pmr::vector<int>, std::pmr::vector<int>>;
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::vector<M> vm(&mr);
    std::pmr::vector<int> ks = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    std::pmr::vector<int> vs = {1, 1, 1, 2, 2, 3, 2, 3, 3};
    assert(ks.get_allocator().resource() != &mr);
    assert(vs.get_allocator().resource() != &mr);
    vm.emplace_back(std::move(ks), std::move(vs));
    LIBCPP_ASSERT(ks.size() == 9); // ks' size is unchanged, since it uses a different allocator
    LIBCPP_ASSERT(vs.size() == 9); // vs' size is unchanged, since it uses a different allocator
    assert((vm[0] == M{{1, 1}, {2, 2}, {3, 3}}));
    assert(vm[0].keys().get_allocator().resource() == &mr);
    assert(vm[0].values().get_allocator().resource() == &mr);
  }

#if 0
  // Test all combinations of lvalue and rvalue containers (LWG 3802).
  {
    int input[] = {1,1,1,2,2,3,2,3,3};
    const P expected[] = {{1,1}, {2,2}, {3,3}};
    {
      using M = std::flat_map<int, MoveOnly, std::less<>, std::pmr::vector<int>, std::pmr::vector<MoveOnly>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<int> ks(input, input + 9);
      std::pmr::vector<MoveOnly> vs(input, input + 9);
      vm.emplace_back(ks, std::move(vs)); // ill-formed before LWG 3802
      assert(ks.size() == 9);        // ks' value is unchanged, since it was an lvalue above
      LIBCPP_ASSERT(vs.size() == 9); // vs' size is unchanged, since it uses a different allocator
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_map<MoveOnly, int, std::less<>, std::pmr::vector<MoveOnly>, std::pmr::vector<int>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<MoveOnly> ks(input, input + 9);
      std::pmr::vector<int> vs(input, input + 9);
      vm.emplace_back(std::move(ks), vs); // ill-formed before LWG 3802
      LIBCPP_ASSERT(ks.size() == 9); // ks' size is unchanged, since it uses a different allocator
      assert(vs.size() == 9);        // vs' value is unchanged, since it was an lvalue above
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_map<MoveOnly, MoveOnly, std::less<>, std::pmr::vector<MoveOnly>, std::pmr::vector<MoveOnly>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<MoveOnly> ks(input, input + 9);
      std::pmr::vector<MoveOnly> vs(input, input + 9);
      vm.emplace_back(std::move(ks), std::move(vs)); // ill-formed before LWG 3802
      LIBCPP_ASSERT(ks.size() == 9); // ks' size is unchanged, since it uses a different allocator
      LIBCPP_ASSERT(vs.size() == 9); // vs' size is unchanged, since it uses a different allocator
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
  }
  {
    int input[] = {1,2,3};
    const P expected[] = {{1,1}, {2,2}, {3,3}};
    {
      using M = std::flat_map<int, MoveOnly, std::less<>, std::pmr::vector<int>, std::pmr::vector<MoveOnly>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<int> ks(input, input + 3);
      std::pmr::vector<MoveOnly> vs(input, input + 3);
      vm.emplace_back(std::sorted_unique, ks, std::move(vs)); // ill-formed before LWG 3802
      assert(ks.size() == 3);        // ks' value is unchanged, since it was an lvalue above
      LIBCPP_ASSERT(vs.size() == 3); // vs' size is unchanged, since it uses a different allocator
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_map<MoveOnly, int, std::less<>, std::pmr::vector<MoveOnly>, std::pmr::vector<int>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<MoveOnly> ks(input, input + 3);
      std::pmr::vector<int> vs(input, input + 3);
      vm.emplace_back(std::sorted_unique, std::move(ks), vs); // ill-formed before LWG 3802
      LIBCPP_ASSERT(ks.size() == 3); // ks' size is unchanged, since it uses a different allocator
      assert(vs.size() == 3);        // vs' value is unchanged, since it was an lvalue above
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
    {
      using M = std::flat_map<MoveOnly, MoveOnly, std::less<>, std::pmr::vector<MoveOnly>, std::pmr::vector<MoveOnly>>;
      std::pmr::monotonic_buffer_resource mr;
      std::pmr::vector<M> vm(&mr);
      std::pmr::vector<MoveOnly> ks(input, input + 3);
      std::pmr::vector<MoveOnly> vs(input, input + 3);
      vm.emplace_back(std::sorted_unique, std::move(ks), std::move(vs)); // ill-formed before LWG 3802
      LIBCPP_ASSERT(ks.size() == 3); // ks' size is unchanged, since it uses a different allocator
      LIBCPP_ASSERT(vs.size() == 3); // vs' size is unchanged, since it uses a different allocator
      assert(std::ranges::equal(vm[0], expected, std::equal_to<>()));
      assert(vm[0].keys().get_allocator().resource() == &mr);
      assert(vm[0].values().get_allocator().resource() == &mr);
    }
  }
#endif
  return 0;
}
