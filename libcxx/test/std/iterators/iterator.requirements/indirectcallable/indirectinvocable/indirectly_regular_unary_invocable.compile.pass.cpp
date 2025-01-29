//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class I>
// concept indirectly_regular_unary_invocable;

#include <iterator>
#include <concepts>

#include "indirectly_readable.h"
#include "test_macros.h"

using It = IndirectlyReadable<struct Token>;
using R1 = T1<struct ReturnToken>;
using R2 = T2<struct ReturnToken>;

template <class I>
struct GoodInvocable {
    R1 operator()(std::iter_value_t<I>&) const;
    R2 operator()(std::iter_reference_t<I>) const;
    R2 operator()(std::iter_common_reference_t<I>) const;
};

// Should work when all constraints are satisfied
static_assert(std::indirectly_regular_unary_invocable<GoodInvocable<It>, It>);

// Should fail when the iterator is not indirectly_readable
struct NotIndirectlyReadable { };
static_assert(!std::indirectly_regular_unary_invocable<GoodInvocable<NotIndirectlyReadable>, NotIndirectlyReadable>);

// Should fail when the invocable is not copy constructible
struct BadInvocable1 {
    BadInvocable1(BadInvocable1 const&) = delete;
    template <class T> R1 operator()(T const&) const;
};
static_assert(!std::indirectly_regular_unary_invocable<BadInvocable1, It>);

// Should fail when the invocable can't be called with (iter_value_t&)
struct BadInvocable2 {
    template <class T> R1 operator()(T const&) const;
    R1 operator()(std::iter_value_t<It>&) const = delete;
};
static_assert(!std::indirectly_regular_unary_invocable<BadInvocable2, It>);

// Should fail when the invocable can't be called with (iter_reference_t)
struct BadInvocable3 {
    template <class T> R1 operator()(T const&) const;
    R1 operator()(std::iter_reference_t<It>) const = delete;
};
static_assert(!std::indirectly_regular_unary_invocable<BadInvocable3, It>);

// This case was made valid by P2997R1.
struct GoodInvocable4 {
  template <class T>
  R1 operator()(T const&) const;
  R1 operator()(std::iter_common_reference_t<It>) const = delete;
};
static_assert(std::indirectly_regular_unary_invocable<GoodInvocable4, It>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable5 {
    R1 operator()(std::iter_value_t<It>&) const;
    struct Unrelated { };
    Unrelated operator()(std::iter_reference_t<It>) const;
    R1 operator()(std::iter_common_reference_t<It>) const;
};
static_assert(!std::indirectly_regular_unary_invocable<BadInvocable5, It>);

// Various tests with callables
struct S;
static_assert(std::indirectly_regular_unary_invocable<int (*)(int), int*>);
static_assert(std::indirectly_regular_unary_invocable<int (&)(int), int*>);
static_assert(std::indirectly_regular_unary_invocable<int S::*, S*>);
static_assert(std::indirectly_regular_unary_invocable<int (S::*)(), S*>);
static_assert(std::indirectly_regular_unary_invocable<int (S::*)() const, S*>);
static_assert(std::indirectly_regular_unary_invocable<void(*)(int), int*>);

static_assert(!std::indirectly_regular_unary_invocable<int(int), int*>); // not move constructible
static_assert(!std::indirectly_regular_unary_invocable<int (*)(int*, int*), int*>);
static_assert(!std::indirectly_regular_unary_invocable<int (&)(int*, int*), int*>);
static_assert(!std::indirectly_regular_unary_invocable<int (S::*)(int*), S*>);
static_assert(!std::indirectly_regular_unary_invocable<int (S::*)(int*) const, S*>);

// Test ADL-proofing (P2538R1)
#if TEST_STD_VER >= 26 || defined(_LIBCPP_VERSION)
struct Incomplete;
template<class T> struct Holder { T t; };
struct HolderIncompletePred { bool operator()(Holder<Incomplete>*) const; };
static_assert(std::indirectly_regular_unary_invocable<HolderIncompletePred, Holder<Incomplete>**>);
static_assert(!std::indirectly_regular_unary_invocable<Holder<Incomplete>*, Holder<Incomplete>**>);
#endif
