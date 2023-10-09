//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template<class T, class Alloc, ...>
// constexpr auto uses_allocator_construction_args(const Alloc& alloc, ...) noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

#include <concepts>
#include <memory>
#include <ranges>
#include <tuple>
#include <utility>

#include "common.h"
#include "test_allocator.h"

template <class Type, class... Args>
constexpr decltype(auto) test_uses_allocator_construction_args(Args&&... args) {
  static_assert(noexcept(std::uses_allocator_construction_args<Type>(std::forward<Args>(args)...)));
  return std::uses_allocator_construction_args<Type>(std::forward<Args>(args)...);
}

template <template <class> class CV>
constexpr void testOne() {
  Alloc a(12);
  {
    std::same_as<std::tuple<std::allocator_arg_t, const Alloc&>> auto ret =
        test_uses_allocator_construction_args<CV<UsesAllocArgT>>(a);
    assert(std::get<1>(ret).get_data() == 12);
  }
  {
    std::same_as<std::tuple<const Alloc&>> auto ret = test_uses_allocator_construction_args<CV<UsesAllocLast>>(a);
    assert(std::get<0>(ret).get_data() == 12);
  }
  {
    [[maybe_unused]] std::same_as<std::tuple<>> auto ret =
        test_uses_allocator_construction_args<CV<NotAllocatorAware>>(a);
  }
  {
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&>,
                            std::tuple<const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(
            a, std::piecewise_construct, std::tuple<>{}, std::tuple<>{});
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<0>(std::get<2>(ret)).get_data() == 12);
  }
  {
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&>,
                            std::tuple<const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(a);
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<0>(std::get<2>(ret)).get_data() == 12);
  }
  {
    int val = 0;
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, int&>,
                            std::tuple<int&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(a, val, val);
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(&std::get<2>(std::get<1>(ret)) == &val);
    assert(&std::get<0>(std::get<2>(ret)) == &val);
  }
  {
    int val = 0;
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, int&&>,
                            std::tuple<int&&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(
            a, std::move(val), std::move(val));
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(&std::get<2>(std::get<1>(ret)) == &val);
    assert(&std::get<0>(std::get<2>(ret)) == &val);
  }
#if TEST_STD_VER >= 23
  {
    std::pair p{3, 4};

    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, int&>,
                            std::tuple<int&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(a, p);
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(std::get<2>(std::get<1>(ret)) == 3);
    assert(std::get<0>(std::get<2>(ret)) == 4);
  }
#endif
  {
    std::pair p{3, 4};
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, const int&>,
                            std::tuple<const int&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(a, std::as_const(p));
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(std::get<2>(std::get<1>(ret)) == 3);
    assert(std::get<0>(std::get<2>(ret)) == 4);
  }
  {
    std::pair p{3, 4};
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, int&&>,
                            std::tuple<int&&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(a, std::move(p));
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(std::get<2>(std::get<1>(ret)) == 3);
    assert(std::get<0>(std::get<2>(ret)) == 4);
  }
#if TEST_STD_VER >= 23
  {
    std::pair p{3, 4};
    std::same_as<std::tuple<std::piecewise_construct_t,
                            std::tuple<std::allocator_arg_t, const Alloc&, const int&&>,
                            std::tuple<const int&&, const Alloc&>>> auto ret =
        test_uses_allocator_construction_args<CV<std::pair<UsesAllocArgT, UsesAllocLast>>>(
            a, std::move(std::as_const(p)));
    assert(std::get<1>(std::get<1>(ret)).get_data() == 12);
    assert(std::get<1>(std::get<2>(ret)).get_data() == 12);
    assert(std::get<2>(std::get<1>(ret)) == 3);
    assert(std::get<0>(std::get<2>(ret)) == 4);
  }
#endif
  {
    ConvertibleToPair ctp{};
    auto ret              = test_uses_allocator_construction_args<CV<std::pair<int, int>>>(a, ctp);
    std::pair<int, int> v = std::get<0>(ret);
    assert(std::get<0>(v) == 1);
    assert(std::get<1>(v) == 2);
  }
  {
    ConvertibleToPair ctp{};
    auto ret              = test_uses_allocator_construction_args<CV<std::pair<int, int>>>(a, std::move(ctp));
    std::pair<int, int> v = std::get<0>(ret);
    assert(std::get<0>(v) == 1);
    assert(std::get<1>(v) == 2);
  }
#if TEST_STD_VER >= 23
  // P2165R4 with LWG3821 applied
  // uses_allocator_construction_args should have overload for pair-like
  {
    // pair-like with explicit ctr should work
    struct Foo {
      int i = 5;
    };
    struct Bar {
      int i;
      constexpr explicit Bar(Foo foo) : i(foo.i) {}
    };

    std::tuple<Foo, Foo> pair_like;
    auto ret  = test_uses_allocator_construction_args<CV<std::pair<Bar, Bar>>>(a, pair_like);
    auto pair = std::make_from_tuple<std::pair<Bar, Bar>>(std::move(ret));
    assert(pair.first.i == 5);
    assert(pair.second.i == 5);
  }
  {
    // subrange should work
    int i = 5;
    std::ranges::subrange<int*, int*> r{&i, &i};
    auto ret  = test_uses_allocator_construction_args<CV<std::pair<int*, int*>>>(a, r);
    auto pair = std::make_from_tuple<std::pair<int*, int*>>(std::move(ret));
    assert(pair.first == &i);
    assert(pair.second == &i);
  }
#endif
}

constexpr bool test() {
  testOne<std::type_identity_t>();

  // LWG 3677. Is a cv-qualified pair specially handled in uses-allocator construction
  testOne<std::add_const_t>();
  testOne<std::add_volatile_t>();
  testOne<std::add_cv_t>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}
