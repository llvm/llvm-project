//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
// concept viewable_range;

#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "test_range.h"

// The constraints we have in viewable_range are:
//  range<T>
//  view<remove_cvref_t<T>>
//  constructible_from<remove_cvref_t<T>, T>
//  lvalue_reference_t<T> || movable<remove_reference_t<T>>
//  is-initializer-list<T>
//
// We test all the relevant combinations of satisfying/not satisfying those constraints.

// viewable_range<T> is not satisfied for (range=false, view=*, constructible_from=*, lvalue-or-movable=*)
struct T1 { };
static_assert(!std::ranges::range<T1>);

static_assert(!std::ranges::viewable_range<T1>);
static_assert(!std::ranges::viewable_range<T1&>);
static_assert(!std::ranges::viewable_range<T1&&>);
static_assert(!std::ranges::viewable_range<T1 const>);
static_assert(!std::ranges::viewable_range<T1 const&>);
static_assert(!std::ranges::viewable_range<T1 const&&>);

// viewable_range<T> is satisfied for (range=true, view=true, constructible_from=true, lvalue-or-movable=true)
struct T2 : test_range<cpp20_input_iterator>, std::ranges::view_base {
  T2(T2 const&) = default;
};
static_assert(std::ranges::range<T2>);
static_assert(std::ranges::view<T2>);
static_assert(std::constructible_from<T2, T2>);

static_assert(std::ranges::viewable_range<T2>);
static_assert(std::ranges::viewable_range<T2&>);
static_assert(std::ranges::viewable_range<T2&&>);
static_assert(std::ranges::viewable_range<T2 const>);
static_assert(std::ranges::viewable_range<T2 const&>);
static_assert(std::ranges::viewable_range<T2 const&&>);

// viewable_range<T> is satisfied for (range=true, view=true, constructible_from=true, lvalue-or-movable=false)
struct T3 : test_range<cpp20_input_iterator>, std::ranges::view_base {
  T3(T3 const&) = default;
};
static_assert(std::ranges::range<T3>);
static_assert(std::ranges::view<T3>);
static_assert(std::constructible_from<T3, T3>);

static_assert(std::ranges::viewable_range<T3>);
static_assert(std::ranges::viewable_range<T3&>);
static_assert(std::ranges::viewable_range<T3&&>);
static_assert(std::ranges::viewable_range<T3 const>);
static_assert(std::ranges::viewable_range<T3 const&>);
static_assert(std::ranges::viewable_range<T3 const&&>);

// viewable_range<T> is not satisfied for (range=true, view=true, constructible_from=false, lvalue-or-movable=true)
struct T4 : test_range<cpp20_input_iterator>, std::ranges::view_base {
  T4(T4 const&) = delete;
  T4(T4&&) = default;             // necessary to model view
  T4& operator=(T4&&) = default;  // necessary to model view
};
static_assert(std::ranges::range<T4 const&>);
static_assert(std::ranges::view<std::remove_cvref_t<T4 const&>>);
static_assert(!std::constructible_from<std::remove_cvref_t<T4 const&>, T4 const&>);

static_assert(!std::ranges::viewable_range<T4 const&>);

// A type that satisfies (range=true, view=true, constructible_from=false, lvalue-or-movable=false) can't be formed,
// because views are movable by definition

// viewable_range<T> is satisfied for (range=true, view=false, constructible_from=true, lvalue-or-movable=true)...
struct T5 : test_range<cpp20_input_iterator> { };
static_assert( std::ranges::range<T5>);
static_assert(!std::ranges::view<T5>);
static_assert( std::constructible_from<T5, T5>);
static_assert( std::movable<T5>);
static_assert(!std::movable<const T5>);

static_assert( std::ranges::viewable_range<T5>); // movable
static_assert( std::ranges::viewable_range<T5&>); // movable
static_assert( std::ranges::viewable_range<T5&&>); // movable
static_assert(!std::ranges::viewable_range<const T5>);
static_assert( std::ranges::viewable_range<const T5&>); // lvalue
static_assert(!std::ranges::viewable_range<const T5&&>);

// ...but not if the (non-view, lvalue-or-movable) range is an initializer_list.
static_assert( std::ranges::range<std::initializer_list<int>>);
static_assert(!std::ranges::view<std::initializer_list<int>>);
static_assert( std::constructible_from<std::initializer_list<int>, std::initializer_list<int>>);
static_assert( std::movable<std::initializer_list<int>>);

static_assert(!std::ranges::viewable_range<std::initializer_list<int>>);
static_assert( std::ranges::viewable_range<std::initializer_list<int>&>);
static_assert(!std::ranges::viewable_range<std::initializer_list<int>&&>);
static_assert(!std::ranges::viewable_range<std::initializer_list<int> const>);
static_assert( std::ranges::viewable_range<std::initializer_list<int> const&>);
static_assert(!std::ranges::viewable_range<std::initializer_list<int> const&&>);

// viewable_range<T> is not satisfied for (range=true, view=false, constructible_from=true, lvalue-or-movable=false)
struct T6 : test_range<cpp20_input_iterator> { T6(T6&&); T6& operator=(T6&&) = delete; };
static_assert( std::ranges::range<T6>);
static_assert(!std::ranges::view<T6>);
static_assert( std::constructible_from<T6, T6>);
static_assert(!std::movable<T6>);

static_assert(!std::ranges::viewable_range<T6>);
static_assert( std::ranges::viewable_range<T6&>); // lvalue
static_assert(!std::ranges::viewable_range<T6&&>);
static_assert(!std::ranges::viewable_range<const T6>);
static_assert( std::ranges::viewable_range<const T6&>); // lvalue
static_assert(!std::ranges::viewable_range<const T6&&>);

// viewable_range<T> is satisfied for (range=true, view=false, constructible_from=false, lvalue-or-movable=true)
struct T7 : test_range<cpp20_input_iterator> {
  T7(T7 const&) = delete;
};
static_assert(std::ranges::range<T7&>);
static_assert(!std::ranges::view<std::remove_cvref_t<T7&>>);
static_assert(!std::constructible_from<std::remove_cvref_t<T7&>, T7&>);

static_assert(!std::ranges::viewable_range<T7>);
static_assert( std::ranges::viewable_range<T7&>);
static_assert(!std::ranges::viewable_range<T7&&>);
static_assert(!std::ranges::viewable_range<const T7>);
static_assert( std::ranges::viewable_range<const T7&>);
static_assert(!std::ranges::viewable_range<const T7&&>);

// viewable_range<T> is not satisfied for (range=true, view=false, constructible_from=false, lvalue-or-movable=false)
struct T8 : test_range<cpp20_input_iterator> {
  T8(T8 const&) = delete;
};
static_assert(std::ranges::range<T8>);
static_assert(!std::ranges::view<T8>);
static_assert(!std::constructible_from<T8, T8>);

static_assert(!std::ranges::viewable_range<T8>);
static_assert( std::ranges::viewable_range<T8&>);
static_assert(!std::ranges::viewable_range<T8&&>);
static_assert(!std::ranges::viewable_range<const T8>);
static_assert( std::ranges::viewable_range<const T8&>);
static_assert(!std::ranges::viewable_range<const T8&&>);

// Test with a few degenerate types
static_assert(!std::ranges::viewable_range<void>);
static_assert(!std::ranges::viewable_range<int>);
static_assert(!std::ranges::viewable_range<int (*)(char)>);
static_assert(!std::ranges::viewable_range<int[]>);
static_assert(!std::ranges::viewable_range<int[10]>);
static_assert(!std::ranges::viewable_range<int(&)[]>); // not a range
static_assert( std::ranges::viewable_range<int(&)[10]>); // OK, lvalue
static_assert(!std::ranges::viewable_range<int(&&)[]>);
static_assert(!std::ranges::viewable_range<int(&&)[10]>);

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };

static_assert(!std::ranges::viewable_range<Holder<Incomplete>*>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>*&>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>*&&>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>* const>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>* const&>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>* const&&>);

static_assert(!std::ranges::viewable_range<Holder<Incomplete>*[10]>);
static_assert( std::ranges::viewable_range<Holder<Incomplete>*(&)[10]>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>*(&&)[10]>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>* const[10]>);
static_assert( std::ranges::viewable_range<Holder<Incomplete>* const(&)[10]>);
static_assert(!std::ranges::viewable_range<Holder<Incomplete>* const(&&)[10]>);
