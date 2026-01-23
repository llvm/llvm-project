//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  template <class O, class T>
//    struct out_value_result;

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "MoveOnly.h"

using std::ranges::out_value_result;

//
// Helper structs
//

// only explicit construction
struct IterTypeExplicit {
  explicit IterTypeExplicit(int*);
};

// implicit construction
struct IterTypeImplicit {
  IterTypeImplicit(int*);
};

struct IterTypeImplicitRef {
  IterTypeImplicitRef(int&);
};

struct NotConvertible {};

template <class T>
struct ConvertibleFrom {
  constexpr ConvertibleFrom(T c) : content{c} {}
  T content;
};

// Standard layout classes can't have virtual functions
struct NonStandardLayoutTypeBase {
  virtual ~NonStandardLayoutTypeBase();
};
struct NonStandardLayoutType : public NonStandardLayoutTypeBase {};

//
constexpr bool test_constraints() {
  // requires convertible_to<const _OutIter1&, _OutIter2> && convertible_to<const _ValType1&, _ValType2>
  static_assert(std::is_constructible_v<out_value_result<int*, int>, out_value_result<int*, int>>);

  // test failure when implicit conversion isn't allowed
  static_assert(!std::is_constructible_v<out_value_result<IterTypeExplicit, int>, out_value_result<int*, int>>);

  // test success when implicit conversion is allowed, checking combinations of value, reference, and const
  static_assert(std::is_constructible_v<out_value_result<IterTypeImplicit, int>, out_value_result<int*, int>>);
  static_assert(std::is_constructible_v<out_value_result<IterTypeImplicit, int>, out_value_result<int*, int> const>);
  static_assert(std::is_constructible_v<out_value_result<IterTypeImplicit, int>, out_value_result<int*, int>&>);
  static_assert(std::is_constructible_v<out_value_result<IterTypeImplicit, int>, out_value_result<int*, int> const&>);

  static_assert(!std::is_constructible_v<out_value_result<IterTypeImplicitRef, int>, out_value_result<int, int>&>);

  // has to be convertible via const&
  static_assert(std::is_convertible_v<out_value_result<int, int>&, out_value_result<long, long>>);
  static_assert(std::is_convertible_v<const out_value_result<int, int>&, out_value_result<long, long>>);
  static_assert(std::is_convertible_v<out_value_result<int, int>&&, out_value_result<long, long>>);
  static_assert(std::is_convertible_v<const out_value_result<int, int>&&, out_value_result<long, long>>);

  // should be move constructible
  static_assert(std::is_move_constructible_v<out_value_result<MoveOnly, int>>);
  static_assert(std::is_move_constructible_v<out_value_result<int, MoveOnly>>);

  // conversions should not work if there is no conversion
  static_assert(!std::is_convertible_v<out_value_result<NotConvertible, int>, out_value_result<int, NotConvertible>>);
  static_assert(!std::is_convertible_v<out_value_result<int, NotConvertible>, out_value_result<NotConvertible, int>>);

  // check standard layout
  static_assert(std::is_standard_layout_v<out_value_result<int, int>>);
  static_assert(!std::is_standard_layout_v<out_value_result<NonStandardLayoutType, int>>);

  return true;
}

// Test results
constexpr bool test() {
  {
    // Check that conversion operator works
    out_value_result<double, int> res{10, 1};
    assert(res.out == 10);
    assert(res.value == 1);
    out_value_result<ConvertibleFrom<double>, ConvertibleFrom<int>> res2 = res;
    assert(res2.out.content == 10);
    assert(res2.value.content == 1);
  }
  {
    // Check that out_value_result isn't overconstrained w.r.t. move/copy constructors
    out_value_result<MoveOnly, int> res{MoveOnly{}, 10};
    assert(res.out.get() == 1);
    assert(res.value == 10);
    auto res2 = std::move(res);
    assert(res.out.get() == 0);
    assert(res.value == 10);
    assert(res2.out.get() == 1);
    assert(res2.value == 10);
  }
  {
    // Check structured binding
    auto [out, val] = out_value_result<int, int>{1, 2};
    assert(out == 1);
    assert(val == 2);
  }
  {
    // Check default construction
    out_value_result<int, double> res;
    static_assert(std::is_same_v<int, decltype(res.out)>);
    static_assert(std::is_same_v<double, decltype(res.value)>);
  }
  {
    // Check aggregate initialization
    out_value_result<int, int> res = {1, 2};
    assert(res.out == 1);
    assert(res.value == 2);
  }

  return true;
}

int main(int, char**) {
  test_constraints();
  static_assert(test_constraints());
  test();
  static_assert(test());
  return 0;
}
