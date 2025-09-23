//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
//
// common_reference specializations for reference_wrapper

#include <concepts>
#include <functional>
#include <type_traits>

template <class T>
concept HasType = requires { typename T::type; };

template <class Result, class T1, class T2>
concept check_XY = std::same_as<Result, std::common_reference_t<T1, T2>>;

template <class Result, class T1, class T2>
concept check_YX = std::same_as<Result, std::common_reference_t<T2, T1>>;

template <class Result, class T1, class T2>
concept check = check_XY<Result, T1, T2> && check_YX<Result, T1, T2>;

template <class T1, class T2>
concept check_none_XY = !HasType<std::common_reference<T1, T2>>;
template <class T1, class T2>
concept check_none_YX = !HasType<std::common_reference<T2, T1>>;

template <class T1, class T2>
concept check_none = check_none_XY<T1, T2> && check_none_YX<T1, T2>;

// https://eel.is/c++draft/meta.trans#other-2.4
template <class X, class Y>
using CondRes = decltype(false ? std::declval<X (&)()>()() : std::declval<Y (&)()>()());

template <class X, class Y>
struct Ternary {};

template <class X, class Y>
  requires requires() { typename CondRes<X, Y>; }
struct Ternary<X, Y> {
  using type = CondRes<X, Y>;
};
template <class X, class Y>
using Ternary_t = typename Ternary<X, Y>::type;

template <class T>
using Ref = std::reference_wrapper<T>;

using std::common_reference_t;
using std::same_as;

// clang-format off
static_assert(check<int &     , Ref<int      >, int &      >);
static_assert(check<int const&, Ref<int      >, int const& >);
static_assert(check<int const&, Ref<int const>, int &      >);
static_assert(check<int const&, Ref<int const>, int const& >);
static_assert(check<int&,       Ref<int> const&, int& >);
static_assert(check<const volatile int&, Ref<const volatile int>, const volatile int&>);

// derived-base and implicit convertibles
struct B {};
struct D : B {};
struct C {
    operator B&() const;
};

static_assert(check<B&      , Ref<B>,       D &     >);
static_assert(check<B const&, Ref<B>,       D const&>);
static_assert(check<B const&, Ref<B const>, D const&>);

static_assert(check<B&      , Ref<D>,       B &     >);
static_assert(check<B const&, Ref<D>,       B const&>);
static_assert(check<B const&, Ref<D const>, B const&>);

static_assert(std::same_as<B&,       CondRes<Ref<D>,       B&>>);
static_assert(std::same_as<B const&, CondRes<Ref<D>,       B const &>>);
static_assert(std::same_as<B const&, CondRes<Ref<D const>, B const&>>);

static_assert( check<B&        , Ref<B>      , C&      >);
static_assert( check<B&        , Ref<B>      , C       >);
static_assert( check<B const&  , Ref<B const>, C       >);
static_assert(!check<B&        , Ref<C>      , B&      >); // Ref<C> cannot be converted to B&
static_assert( check<B&        , Ref<B>      , C const&>); // was const B& before P2655R3


using Ri   = Ref<int>;
using RRi  = Ref<Ref<int>>;
using RRRi = Ref<Ref<Ref<int>>>;
static_assert(check<Ri&,  Ri&,  RRi>);
static_assert(check<RRi&, RRi&, RRRi>);
static_assert(check<Ri,   Ri,   RRi>);
static_assert(check<RRi,  RRi,  RRRi>);

static_assert(check_none<int&, RRi>);
static_assert(check_none<int,  RRi>);
static_assert(check_none<int&, RRRi>);
static_assert(check_none<int,  RRRi>);

static_assert(check_none<Ri&, RRRi>);
static_assert(check_none<Ri,  RRRi>);


template <typename T>
struct Test {
  // Check that reference_wrapper<T> behaves the same as T& in common_reference.

  using R1 = common_reference_t<T&, T&>;
  using R2 = common_reference_t<T&, T const&>;
  using R3 = common_reference_t<T&, T&&>;
  using R4 = common_reference_t<T&, T const&&>;
  using R5 = common_reference_t<T&, T>;

  static_assert(same_as<R1, common_reference_t<Ref<T>, T&>>);
  static_assert(same_as<R2, common_reference_t<Ref<T>, T const&>>);
  static_assert(same_as<R3, common_reference_t<Ref<T>, T&&>>);
  static_assert(same_as<R4, common_reference_t<Ref<T>, T const&&>>);
  static_assert(same_as<R5, common_reference_t<Ref<T>, T>>);

  // commute:
  static_assert(same_as<R1, common_reference_t<T&,        Ref<T>>>);
  static_assert(same_as<R2, common_reference_t<T const&,  Ref<T>>>);
  static_assert(same_as<R3, common_reference_t<T&&,       Ref<T>>>);
  static_assert(same_as<R4, common_reference_t<T const&&, Ref<T>>>);
  static_assert(same_as<R5, common_reference_t<T,         Ref<T>>>);

  // reference qualification of reference_wrapper is irrelevant 
  static_assert(same_as<R1, common_reference_t<Ref<T>&,        T&>>);
  static_assert(same_as<R1, common_reference_t<Ref<T> ,        T&>>);
  static_assert(same_as<R1, common_reference_t<Ref<T> const&,  T&>>);
  static_assert(same_as<R1, common_reference_t<Ref<T>&&,       T&>>);
  static_assert(same_as<R1, common_reference_t<Ref<T> const&&, T&>>);
};

// clang-format on
// Instantiate above checks:
template struct Test<int>;
template struct Test<std::reference_wrapper<int>>;

// reference_wrapper as both args is unaffected.
// subject to simple first rule of
static_assert(check<Ref<int>&, Ref<int>&, Ref<int>&>);

// double wrap is unaffected.
static_assert(check<Ref<int>&, Ref<Ref<int>>, Ref<int>&>);
