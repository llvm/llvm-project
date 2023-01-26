//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types>
// template <class... UTypes>
//   constexpr explicit(see below) tuple<Types>::tuple(tuple<UTypes...>&);
//
// Constraints:
//  sizeof...(Types) equals sizeof...(UTypes) &&
//  (is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...) is true &&
//  (
//    sizeof...(Types) is not 1 ||
//    (
//      !is_convertible_v<decltype(u), T> &&
//      !is_constructible_v<T, decltype(u)> &&
//      !is_same_v<T, U>
//    )
//  )

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <tuple>

#include "copy_move_types.h"
#include "test_macros.h"

// test: The expression inside explicit is equivalent to:
// !(is_convertible_v<decltype(get<I>(FWD(u))), Types> && ...)
static_assert(std::is_convertible_v<std::tuple<MutableCopy>&, std::tuple<ConvertibleFrom<MutableCopy>>>);

static_assert(std::is_convertible_v<std::tuple<MutableCopy, MutableCopy>&,
                                    std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<std::tuple<MutableCopy>&, std::tuple<ExplicitConstructibleFrom<MutableCopy>>>);

static_assert(!std::is_convertible_v<std::tuple<MutableCopy, MutableCopy>&,
                                     std::tuple<ConvertibleFrom<MutableCopy>, ExplicitConstructibleFrom<MutableCopy>>>);

constexpr bool test() {
  // test implicit conversions.
  // sizeof...(Types) == 1
  {
    std::tuple<MutableCopy> t1{1};
    std::tuple<ConvertibleFrom<MutableCopy>> t2 = t1;
    assert(std::get<0>(t2).v.val == 1);
  }

  // test implicit conversions.
  // sizeof...(Types) > 1
  {
    std::tuple<MutableCopy, int> t1{1, 2};
    std::tuple<ConvertibleFrom<MutableCopy>, int> t2 = t1;
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<1>(t2) == 2);
  }

  // test explicit conversions.
  // sizeof...(Types) == 1
  {
    std::tuple<MutableCopy> t1{1};
    std::tuple<ExplicitConstructibleFrom<MutableCopy>> t2{t1};
    assert(std::get<0>(t2).v.val == 1);
  }

  // test explicit conversions.
  // sizeof...(Types) > 1
  {
    std::tuple<MutableCopy, int> t1{1, 2};
    std::tuple<ExplicitConstructibleFrom<MutableCopy>, int> t2{t1};
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<1>(t2) == 2);
  }

  // test constraints

  // sizeof...(Types) != sizeof...(UTypes)
  static_assert(!std::is_constructible_v<std::tuple<int, int>, std::tuple<int>&>);
  static_assert(!std::is_constructible_v<std::tuple<int>, std::tuple<int, int>&>);
  static_assert(!std::is_constructible_v<std::tuple<int, int, int>, std::tuple<int, int>&>);

  // !(is_constructible_v<Types, decltype(get<I>(FWD(u)))> && ...)
  static_assert(!std::is_constructible_v<std::tuple<int, NoConstructorFromInt>, std::tuple<int, int>&>);

  // sizeof...(Types) == 1 && other branch of "||" satisfied
  {
    std::tuple<TracedCopyMove> t1{};
    std::tuple<ConvertibleFrom<TracedCopyMove>> t2{t1};
    assert(nonConstCopyCtrCalled(std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_same_v<T, U>
  {
    std::tuple<TracedCopyMove> t1{};
    std::tuple<TracedCopyMove> t2{t1};
    assert(!nonConstCopyCtrCalled(std::get<0>(t2)));
  }

  // sizeof...(Types) != 1
  {
    std::tuple<TracedCopyMove, TracedCopyMove> t1{};
    std::tuple<TracedCopyMove, TracedCopyMove> t2{t1};
    assert(nonConstCopyCtrCalled(std::get<0>(t2)));
  }

  // These two test points cause gcc to ICE
#if !defined(TEST_COMPILER_GCC)
  // sizeof...(Types) == 1 && is_convertible_v<decltype(u), T>
  {
    std::tuple<CvtFromTupleRef> t1{};
    std::tuple<ConvertibleFrom<CvtFromTupleRef>> t2{t1};
    assert(!nonConstCopyCtrCalled(std::get<0>(t2).v));
  }

  // sizeof...(Types) == 1 && is_constructible_v<decltype(u), T>
  {
    std::tuple<ExplicitCtrFromTupleRef> t1{};
    std::tuple<ConvertibleFrom<ExplicitCtrFromTupleRef>> t2{t1};
    assert(!nonConstCopyCtrCalled(std::get<0>(t2).v));
  }
#endif
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
