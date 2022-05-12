//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types>
// template<class Alloc, class... UTypes>
//   constexpr explicit(see below)
//     tuple<Types...>::tuple(allocator_arg_t, const Alloc& a,
//     const tuple<UTypes...>&&);
//
// Constraints:
//  sizeof...(Types) equals sizeof...(UTypes) &&
//  (is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...) is true &&
//  (
//	sizeof...(Types) is not 1 ||
//	(
//		!is_convertible_v<decltype(u), T> &&
//		!is_constructible_v<T, decltype(u)> &&
//		!is_same_v<T, U>
//	)
//  )

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <tuple>

#include "convert_types.h"
#include "test_allocator.h"
#include "test_macros.h"

// test: The expression inside explicit is equivalent to:
// !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
static_assert(ImplicitlyConstructible< std::tuple<ConvertibleFrom<ConstMove>>, std::allocator_arg_t,
                                       const test_allocator<int>&, const std::tuple<ConstMove>&&>);

static_assert(
    ImplicitlyConstructible< std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>, std::allocator_arg_t,
                             const test_allocator<int>&, const std::tuple<ConstMove, ConstMove>&&>);

static_assert(!ImplicitlyConstructible<std::tuple<ExplicitConstructibleFrom<ConstMove>>, std::allocator_arg_t,
                                       const test_allocator<int>&, const std::tuple<ConstMove>&&>);

static_assert(!ImplicitlyConstructible<std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>>,
                                       std::allocator_arg_t, const test_allocator<int>&,
                                       const std::tuple<ConstMove, ConstMove>&&>);

constexpr bool test() {
  // test implicit conversions.
  // sizeof...(Types) == 1
  {
    const std::tuple<ConstMove> t1{1};
    std::tuple<ConvertibleFrom<ConstMove>> t2 = {std::allocator_arg, test_allocator<int>{}, std::move(t1)};
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<0>(t2).alloc_constructed);
  }

  // test implicit conversions.
  // sizeof...(Types) > 1
  {
    const std::tuple<ConstMove, int> t1{1, 2};
    std::tuple<ConvertibleFrom<ConstMove>, int> t2 = {std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<1>(t2) == 2);
    assert(std::get<0>(t2).alloc_constructed);
  }

  // test explicit conversions.
  // sizeof...(Types) == 1
  {
    const std::tuple<ConstMove> t1{1};
    std::tuple<ExplicitConstructibleFrom<ConstMove>> t2{std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<0>(t2).alloc_constructed);
  }

  // test explicit conversions.
  // sizeof...(Types) > 1
  {
    const std::tuple<ConstMove, int> t1{1, 2};
    std::tuple<ExplicitConstructibleFrom<ConstMove>, int> t2{std::allocator_arg_t{}, test_allocator<int>{},
                                                             std::move(t1)};
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<1>(t2) == 2);
    assert(std::get<0>(t2).alloc_constructed);
  }

  // test constraints

  // sizeof...(Types) != sizeof...(UTypes)
  static_assert(!std::is_constructible_v<std::tuple<int, int>, std::allocator_arg_t, const test_allocator<int>&,
                                         const std::tuple<int>&&>);
  static_assert(!std::is_constructible_v<std::tuple<int, int, int>, std::allocator_arg_t, const test_allocator<int>&,
                                         const std::tuple<int, int>&&>);

  // !(is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...)
  static_assert(!std::is_constructible_v< std::tuple<int, NoConstructorFromInt>, std::allocator_arg_t,
                                          const test_allocator<int>&, const std::tuple<int, int>&&>);

  // sizeof...(Types) == 1 && other branch of "||" satisfied
  {
    const std::tuple<TracedCopyMove> t1{};
    std::tuple<ConvertibleFrom<TracedCopyMove>> t2{std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(constMoveCtrCalled(std::get<0>(t2).v));
    assert(std::get<0>(t2).alloc_constructed);
  }

  // sizeof...(Types) == 1 && is_same_v<T, U>
  {
    const std::tuple<TracedCopyMove> t1{};
    std::tuple<TracedCopyMove> t2{std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(!constMoveCtrCalled(std::get<0>(t2)));
    assert(std::get<0>(t2).alloc_constructed);
  }

  // sizeof...(Types) != 1
  {
    const std::tuple<TracedCopyMove, TracedCopyMove> t1{};
    std::tuple<TracedCopyMove, TracedCopyMove> t2{std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(constMoveCtrCalled(std::get<0>(std::move(t2))));
    assert(std::get<0>(t2).alloc_constructed);
  }

  // These two test points cause gcc to ICE
#if !defined(TEST_COMPILER_GCC)
  // sizeof...(Types) == 1 && is_convertible_v<decltype(u), T>
  {
    const std::tuple<CvtFromTupleRef> t1{};
    std::tuple<ConvertibleFrom<CvtFromTupleRef>> t2{std::allocator_arg_t{}, test_allocator<int>{}, std::move(t1)};
    assert(!constMoveCtrCalled(std::get<0>(t2).v));
    assert(std::get<0>(t2).alloc_constructed);
  }

  // sizeof...(Types) == 1 && is_constructible_v<decltype(u), T>
  {
    const std::tuple<ExplicitCtrFromTupleRef> t1{};
    std::tuple<ConvertibleFrom<ExplicitCtrFromTupleRef>> t2{std::allocator_arg_t{}, test_allocator<int>{},
                                                            std::move(t1)};
    assert(!constMoveCtrCalled(std::get<0>(t2).v));
    assert(std::get<0>(t2).alloc_constructed);
  }
#endif
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
