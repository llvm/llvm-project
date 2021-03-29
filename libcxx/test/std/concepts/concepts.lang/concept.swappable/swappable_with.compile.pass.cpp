//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T, class U>
// concept swappable_with = // see below

#include <concepts>

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "type_classification/moveconstructible.h"
#include "type_classification/swappable.h"

template <class T, class U>
constexpr bool check_swappable_with_impl() {
  static_assert(std::swappable_with<T, U> == std::swappable_with<U, T>);
  return std::swappable_with<T, U>;
}

template <class T, class U>
constexpr bool check_swappable_with() {
  static_assert(!check_swappable_with_impl<T, U>());
  static_assert(!check_swappable_with_impl<T, U const>());
  static_assert(!check_swappable_with_impl<T const, U>());
  static_assert(!check_swappable_with_impl<T const, U const>());

  static_assert(!check_swappable_with_impl<T, U&>());
  static_assert(!check_swappable_with_impl<T, U const&>());
  static_assert(!check_swappable_with_impl<T, U volatile&>());
  static_assert(!check_swappable_with_impl<T, U const volatile&>());
  static_assert(!check_swappable_with_impl<T const, U&>());
  static_assert(!check_swappable_with_impl<T const, U const&>());
  static_assert(!check_swappable_with_impl<T const, U volatile&>());
  static_assert(!check_swappable_with_impl<T const, U const volatile&>());

  static_assert(!check_swappable_with_impl<T&, U>());
  static_assert(!check_swappable_with_impl<T&, U const>());
  static_assert(!check_swappable_with_impl<T const&, U>());
  static_assert(!check_swappable_with_impl<T const&, U const>());
  static_assert(!check_swappable_with_impl<T volatile&, U>());
  static_assert(!check_swappable_with_impl<T volatile&, U const>());
  static_assert(!check_swappable_with_impl<T const volatile&, U>());
  static_assert(!check_swappable_with_impl<T const volatile&, U const>());

  static_assert(!check_swappable_with_impl<T&, U const&>());
  static_assert(!check_swappable_with_impl<T&, U volatile&>());
  static_assert(!check_swappable_with_impl<T&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T const&, U&>());
  static_assert(!check_swappable_with_impl<T const&, U const&>());
  static_assert(!check_swappable_with_impl<T const&, U volatile&>());
  static_assert(!check_swappable_with_impl<T const&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T volatile&, U&>());
  static_assert(!check_swappable_with_impl<T volatile&, U const&>());
  static_assert(!check_swappable_with_impl<T volatile&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U const&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U volatile&>());
  static_assert(
      !check_swappable_with_impl<T const volatile&, U const volatile&>());

  static_assert(!check_swappable_with_impl<T, U&&>());
  static_assert(!check_swappable_with_impl<T, U const&&>());
  static_assert(!check_swappable_with_impl<T, U volatile&&>());
  static_assert(!check_swappable_with_impl<T, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T const, U&&>());
  static_assert(!check_swappable_with_impl<T const, U const&&>());
  static_assert(!check_swappable_with_impl<T const, U volatile&&>());
  static_assert(!check_swappable_with_impl<T const, U const volatile&&>());

  static_assert(!check_swappable_with_impl<T&&, U>());
  static_assert(!check_swappable_with_impl<T&&, U const>());
  static_assert(!check_swappable_with_impl<T const&&, U>());
  static_assert(!check_swappable_with_impl<T const&&, U const>());
  static_assert(!check_swappable_with_impl<T volatile&&, U>());
  static_assert(!check_swappable_with_impl<T volatile&&, U const>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U const>());

  static_assert(!check_swappable_with_impl<T&, U&&>());
  static_assert(!check_swappable_with_impl<T&, U const&&>());
  static_assert(!check_swappable_with_impl<T&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T const&, U&&>());
  static_assert(!check_swappable_with_impl<T const&, U const&&>());
  static_assert(!check_swappable_with_impl<T const&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T const&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T volatile&, U&&>());
  static_assert(!check_swappable_with_impl<T volatile&, U const&&>());
  static_assert(!check_swappable_with_impl<T volatile&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T volatile&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U&&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U const&&>());
  static_assert(!check_swappable_with_impl<T const volatile&, U volatile&&>());
  static_assert(
      !check_swappable_with_impl<T const volatile&, U const volatile&&>());

  static_assert(!check_swappable_with_impl<T&&, U&>());
  static_assert(!check_swappable_with_impl<T&&, U const&>());
  static_assert(!check_swappable_with_impl<T&&, U volatile&>());
  static_assert(!check_swappable_with_impl<T&&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T const&&, U&>());
  static_assert(!check_swappable_with_impl<T const&&, U const&>());
  static_assert(!check_swappable_with_impl<T const&&, U volatile&>());
  static_assert(!check_swappable_with_impl<T const&&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U const&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U volatile&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U const volatile&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U const&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U volatile&>());
  static_assert(
      !check_swappable_with_impl<T const volatile&&, U const volatile&>());

  static_assert(!check_swappable_with_impl<T&&, U&&>());
  static_assert(!check_swappable_with_impl<T&&, U const&&>());
  static_assert(!check_swappable_with_impl<T&&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T&&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T const&&, U&&>());
  static_assert(!check_swappable_with_impl<T const&&, U const&&>());
  static_assert(!check_swappable_with_impl<T const&&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T const&&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U&&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U const&&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U volatile&&>());
  static_assert(!check_swappable_with_impl<T volatile&&, U const volatile&&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U&&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U const&&>());
  static_assert(!check_swappable_with_impl<T const volatile&&, U volatile&&>());
  static_assert(
      !check_swappable_with_impl<T const volatile&&, U const volatile&&>());
  return check_swappable_with_impl<T&, U&>();
}

template <class T, class U>
constexpr bool check_swappable_with_including_lvalue_ref_to_volatile() {
  constexpr auto result = check_swappable_with<T, U>();
  static_assert(check_swappable_with_impl<T volatile&, U volatile&>() ==
                result);
  return result;
}

namespace fundamental {
static_assert(
    check_swappable_with_including_lvalue_ref_to_volatile<int, int>());
static_assert(
    check_swappable_with_including_lvalue_ref_to_volatile<double, double>());
static_assert(
    !check_swappable_with_including_lvalue_ref_to_volatile<int, double>());

static_assert(
    check_swappable_with_including_lvalue_ref_to_volatile<int*, int*>());
static_assert(
    !check_swappable_with_including_lvalue_ref_to_volatile<int, int*>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (*)(), int (*)()>());
static_assert(
    !check_swappable_with_including_lvalue_ref_to_volatile<int, int (*)()>());

struct S {};
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int, S>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int S::*, int S::*>());
static_assert(
    !check_swappable_with_including_lvalue_ref_to_volatile<int, int S::*>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)(), int (S::*)()>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int, int (S::*)()>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() noexcept, int (S::*)() noexcept>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() noexcept, int (S::*)()>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const, int (S::*)() const>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const, int (S::*)()>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const noexcept, int (S::*)() const noexcept>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const, int (S::*)() const noexcept>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() volatile, int (S::*)() volatile>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() volatile, int (S::*)()>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const volatile, int (S::*)() const volatile>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int (S::*)() const volatile, int (S::*)()>());

static_assert(
    check_swappable_with_including_lvalue_ref_to_volatile<int[5], int[5]>());
static_assert(
    !check_swappable_with_including_lvalue_ref_to_volatile<int[5], int[6]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5], double[5]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5], double[6]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6],
                                                                     int[5]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<int[5][6],
                                                                     int[6]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], double[5]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], double[6]>());
static_assert(check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], int[5][6]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], int[5][4]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], int[6][5]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], double[5][6]>());
static_assert(!check_swappable_with_including_lvalue_ref_to_volatile<
              int[5][6], double[6][5]>());

// always false
static_assert(!check_swappable_with_impl<void, void>());
static_assert(!check_swappable_with_impl<int, void>());
static_assert(!check_swappable_with_impl<int&, void>());
static_assert(!check_swappable_with_impl<void, int>());
static_assert(!check_swappable_with_impl<void, int&>());
static_assert(!check_swappable_with_impl<int, int()>());
static_assert(!check_swappable_with_impl<int, int (&)()>());
} // namespace fundamental

namespace adl {
static_assert(
    check_swappable_with<lvalue_adl_swappable, lvalue_adl_swappable>());
static_assert(check_swappable_with<lvalue_rvalue_adl_swappable,
                                   lvalue_rvalue_adl_swappable>());
static_assert(check_swappable_with<rvalue_lvalue_adl_swappable,
                                   rvalue_lvalue_adl_swappable>());
static_assert(
    check_swappable_with_impl<rvalue_adl_swappable, rvalue_adl_swappable>());
static_assert(!check_swappable_with_impl<lvalue_rvalue_adl_swappable&,
                                         lvalue_rvalue_adl_swappable&&>());

struct s1 {};
struct no_common_reference_with_s1 {
  friend void swap(s1&, no_common_reference_with_s1&);
  friend void swap(no_common_reference_with_s1&, s1&);
};
static_assert(!check_swappable_with<s1, no_common_reference_with_s1>());

struct one_way_swappable_with_s1 {
  friend void swap(s1&, one_way_swappable_with_s1&);
  operator s1();
};
static_assert(std::common_reference_with<one_way_swappable_with_s1, s1>);
static_assert(!check_swappable_with<one_way_swappable_with_s1, s1>());

struct one_way_swappable_with_s1_other_way {
  friend void swap(one_way_swappable_with_s1_other_way&, s1&);
  operator s1();
};
static_assert(
    std::common_reference_with<one_way_swappable_with_s1_other_way, s1>);
static_assert(!check_swappable_with<one_way_swappable_with_s1_other_way, s1>());

struct can_swap_with_s1_but_not_swappable {
  can_swap_with_s1_but_not_swappable(can_swap_with_s1_but_not_swappable&&) =
      delete;
  friend void swap(s1&, can_swap_with_s1_but_not_swappable&);
  friend void swap(can_swap_with_s1_but_not_swappable&, s1&);

  operator s1() const;
};
static_assert(
    std::common_reference_with<can_swap_with_s1_but_not_swappable, s1>);
static_assert(!std::swappable<can_swap_with_s1_but_not_swappable>);
static_assert(
    !check_swappable_with<can_swap_with_s1_but_not_swappable&, s1&>());

struct swappable_with_s1 {
  friend void swap(s1&, swappable_with_s1&);
  friend void swap(swappable_with_s1&, s1&);
  operator s1() const;
};
static_assert(check_swappable_with<swappable_with_s1, s1>());

struct swappable_with_const_s1_but_not_swappable {
  swappable_with_const_s1_but_not_swappable(
      swappable_with_const_s1_but_not_swappable const&);
  swappable_with_const_s1_but_not_swappable(
      swappable_with_const_s1_but_not_swappable const&&);
  swappable_with_const_s1_but_not_swappable&
  operator=(swappable_with_const_s1_but_not_swappable const&);
  swappable_with_const_s1_but_not_swappable&
  operator=(swappable_with_const_s1_but_not_swappable const&&);

  friend void swap(s1 const&, swappable_with_const_s1_but_not_swappable const&);
  friend void swap(swappable_with_const_s1_but_not_swappable const&, s1 const&);

  operator s1 const &() const;
};
static_assert(
    !std::swappable<swappable_with_const_s1_but_not_swappable const&>);
static_assert(!std::swappable_with<
              swappable_with_const_s1_but_not_swappable const&, s1 const&>);

struct swappable_with_volatile_s1_but_not_swappable {
  swappable_with_volatile_s1_but_not_swappable(
      swappable_with_volatile_s1_but_not_swappable volatile&);
  swappable_with_volatile_s1_but_not_swappable(
      swappable_with_volatile_s1_but_not_swappable volatile&&);
  swappable_with_volatile_s1_but_not_swappable&
  operator=(swappable_with_volatile_s1_but_not_swappable volatile&);
  swappable_with_volatile_s1_but_not_swappable&
  operator=(swappable_with_volatile_s1_but_not_swappable volatile&&);

  friend void swap(s1 volatile&,
                   swappable_with_volatile_s1_but_not_swappable volatile&);
  friend void swap(swappable_with_volatile_s1_but_not_swappable volatile&,
                   s1 volatile&);

  operator s1 volatile &() volatile;
};
static_assert(
    !std::swappable<swappable_with_volatile_s1_but_not_swappable volatile&>);
static_assert(
    !std::swappable_with<swappable_with_volatile_s1_but_not_swappable volatile&,
                         s1 volatile&>);

struct swappable_with_cv_s1_but_not_swappable {
  swappable_with_cv_s1_but_not_swappable(
      swappable_with_cv_s1_but_not_swappable const volatile&);
  swappable_with_cv_s1_but_not_swappable(
      swappable_with_cv_s1_but_not_swappable const volatile&&);
  swappable_with_cv_s1_but_not_swappable&
  operator=(swappable_with_cv_s1_but_not_swappable const volatile&);
  swappable_with_cv_s1_but_not_swappable&
  operator=(swappable_with_cv_s1_but_not_swappable const volatile&&);

  friend void swap(s1 const volatile&,
                   swappable_with_cv_s1_but_not_swappable const volatile&);
  friend void swap(swappable_with_cv_s1_but_not_swappable const volatile&,
                   s1 const volatile&);

  operator s1 const volatile &() const volatile;
};
static_assert(
    !std::swappable<swappable_with_cv_s1_but_not_swappable const volatile&>);
static_assert(
    !std::swappable_with<swappable_with_cv_s1_but_not_swappable const volatile&,
                         s1 const volatile&>);

struct s2 {
  friend void swap(s2 const&, s2 const&);
  friend void swap(s2 volatile&, s2 volatile&);
  friend void swap(s2 const volatile&, s2 const volatile&);
};

struct swappable_with_const_s2 {
  swappable_with_const_s2(swappable_with_const_s2 const&);
  swappable_with_const_s2(swappable_with_const_s2 const&&);
  swappable_with_const_s2& operator=(swappable_with_const_s2 const&);
  swappable_with_const_s2& operator=(swappable_with_const_s2 const&&);

  friend void swap(swappable_with_const_s2 const&,
                   swappable_with_const_s2 const&);
  friend void swap(s2 const&, swappable_with_const_s2 const&);
  friend void swap(swappable_with_const_s2 const&, s2 const&);

  operator s2 const &() const;
};
static_assert(std::swappable_with<swappable_with_const_s2 const&, s2 const&>);

struct swappable_with_volatile_s2 {
  swappable_with_volatile_s2(swappable_with_volatile_s2 volatile&);
  swappable_with_volatile_s2(swappable_with_volatile_s2 volatile&&);
  swappable_with_volatile_s2& operator=(swappable_with_volatile_s2 volatile&);
  swappable_with_volatile_s2& operator=(swappable_with_volatile_s2 volatile&&);

  friend void swap(swappable_with_volatile_s2 volatile&,
                   swappable_with_volatile_s2 volatile&);
  friend void swap(s2 volatile&, swappable_with_volatile_s2 volatile&);
  friend void swap(swappable_with_volatile_s2 volatile&, s2 volatile&);

  operator s2 volatile &() volatile;
};
static_assert(
    std::swappable_with<swappable_with_volatile_s2 volatile&, s2 volatile&>);

struct swappable_with_cv_s2 {
  swappable_with_cv_s2(swappable_with_cv_s2 const volatile&);
  swappable_with_cv_s2(swappable_with_cv_s2 const volatile&&);
  swappable_with_cv_s2& operator=(swappable_with_cv_s2 const volatile&);
  swappable_with_cv_s2& operator=(swappable_with_cv_s2 const volatile&&);

  friend void swap(swappable_with_cv_s2 const volatile&,
                   swappable_with_cv_s2 const volatile&);
  friend void swap(s2 const volatile&, swappable_with_cv_s2 const volatile&);
  friend void swap(swappable_with_cv_s2 const volatile&, s2 const volatile&);

  operator s2 const volatile &() const volatile;
};
static_assert(std::swappable_with<swappable_with_cv_s2 const volatile&,
                                  s2 const volatile&>);

struct swappable_with_rvalue_ref_to_s1_but_not_swappable {
  friend void swap(swappable_with_rvalue_ref_to_s1_but_not_swappable&&,
                   swappable_with_rvalue_ref_to_s1_but_not_swappable&&);
  friend void swap(s1&&, swappable_with_rvalue_ref_to_s1_but_not_swappable&&);
  friend void swap(swappable_with_rvalue_ref_to_s1_but_not_swappable&&, s1&&);

  operator s1() const;
};
static_assert(
    !std::swappable<swappable_with_rvalue_ref_to_s1_but_not_swappable const&&>);
static_assert(
    !std::swappable_with<
        swappable_with_rvalue_ref_to_s1_but_not_swappable const&&, s1 const&&>);

struct swappable_with_rvalue_ref_to_const_s1_but_not_swappable {
  friend void
  swap(s1 const&&,
       swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&);
  friend void
  swap(swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&,
       s1 const&&);

  operator s1 const() const;
};
static_assert(!std::swappable<
              swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&>);
static_assert(!std::swappable_with<
              swappable_with_rvalue_ref_to_const_s1_but_not_swappable const&&,
              s1 const&&>);

struct swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable {
  swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable(
      swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&);
  swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable(
      swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);
  swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable& operator=(
      swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&);
  swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable& operator=(
      swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);

  friend void
  swap(s1 volatile&&,
       swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&);
  friend void
  swap(swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&,
       s1 volatile&&);

  operator s1 volatile &&() volatile&&;
};
static_assert(
    !std::swappable<
        swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&>);
static_assert(
    !std::swappable_with<
        swappable_with_rvalue_ref_to_volatile_s1_but_not_swappable volatile&&,
        s1 volatile&&>);

struct swappable_with_rvalue_ref_to_cv_s1_but_not_swappable {
  swappable_with_rvalue_ref_to_cv_s1_but_not_swappable(
      swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&);
  swappable_with_rvalue_ref_to_cv_s1_but_not_swappable(
      swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);
  swappable_with_rvalue_ref_to_cv_s1_but_not_swappable& operator=(
      swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&);
  swappable_with_rvalue_ref_to_cv_s1_but_not_swappable& operator=(
      swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);

  friend void
  swap(s1 const volatile&&,
       swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&);
  friend void
  swap(swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&,
       s1 const volatile&&);

  operator s1 const volatile &&() const volatile&&;
};
static_assert(
    !std::swappable<
        swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&>);
static_assert(
    !std::swappable_with<
        swappable_with_rvalue_ref_to_cv_s1_but_not_swappable const volatile&&,
        s1 const volatile&&>);

struct s3 {
  friend void swap(s3&&, s3&&);
  friend void swap(s3 const&&, s3 const&&);
  friend void swap(s3 volatile&&, s3 volatile&&);
  friend void swap(s3 const volatile&&, s3 const volatile&&);
};

struct swappable_with_rvalue_ref_to_s3 {
  friend void swap(swappable_with_rvalue_ref_to_s3&&,
                   swappable_with_rvalue_ref_to_s3&&);
  friend void swap(s3&&, swappable_with_rvalue_ref_to_s3&&);
  friend void swap(swappable_with_rvalue_ref_to_s3&&, s3&&);

  operator s3() const;
};
static_assert(std::swappable_with<swappable_with_rvalue_ref_to_s3&&, s3&&>);

struct swappable_with_rvalue_ref_to_const_s3 {
  swappable_with_rvalue_ref_to_const_s3(
      swappable_with_rvalue_ref_to_const_s3 const&);
  swappable_with_rvalue_ref_to_const_s3(
      swappable_with_rvalue_ref_to_const_s3 const&&);
  swappable_with_rvalue_ref_to_const_s3&
  operator=(swappable_with_rvalue_ref_to_const_s3 const&);
  swappable_with_rvalue_ref_to_const_s3&
  operator=(swappable_with_rvalue_ref_to_const_s3 const&&);

  friend void swap(swappable_with_rvalue_ref_to_const_s3 const&&,
                   swappable_with_rvalue_ref_to_const_s3 const&&);
  friend void swap(s3 const&&, swappable_with_rvalue_ref_to_const_s3 const&&);
  friend void swap(swappable_with_rvalue_ref_to_const_s3 const&&, s3 const&&);

  operator s3() const;
};
static_assert(std::swappable_with<swappable_with_rvalue_ref_to_const_s3 const&&,
                                  s3 const&&>);

struct swappable_with_rvalue_ref_to_volatile_s3 {
  swappable_with_rvalue_ref_to_volatile_s3(
      swappable_with_rvalue_ref_to_volatile_s3 volatile&);
  swappable_with_rvalue_ref_to_volatile_s3(
      swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  swappable_with_rvalue_ref_to_volatile_s3&
  operator=(swappable_with_rvalue_ref_to_volatile_s3 volatile&);
  swappable_with_rvalue_ref_to_volatile_s3&
  operator=(swappable_with_rvalue_ref_to_volatile_s3 volatile&&);

  friend void swap(swappable_with_rvalue_ref_to_volatile_s3 volatile&&,
                   swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  friend void swap(s3 volatile&&,
                   swappable_with_rvalue_ref_to_volatile_s3 volatile&&);
  friend void swap(swappable_with_rvalue_ref_to_volatile_s3 volatile&&,
                   s3 volatile&&);

  operator s3 volatile &() volatile;
};
static_assert(
    std::swappable_with<swappable_with_rvalue_ref_to_volatile_s3 volatile&&,
                        s3 volatile&&>);

struct swappable_with_rvalue_ref_to_cv_s3 {
  swappable_with_rvalue_ref_to_cv_s3(
      swappable_with_rvalue_ref_to_cv_s3 const volatile&);
  swappable_with_rvalue_ref_to_cv_s3(
      swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  swappable_with_rvalue_ref_to_cv_s3&
  operator=(swappable_with_rvalue_ref_to_cv_s3 const volatile&);
  swappable_with_rvalue_ref_to_cv_s3&
  operator=(swappable_with_rvalue_ref_to_cv_s3 const volatile&&);

  friend void swap(swappable_with_rvalue_ref_to_cv_s3 const volatile&&,
                   swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  friend void swap(s3 const volatile&&,
                   swappable_with_rvalue_ref_to_cv_s3 const volatile&&);
  friend void swap(swappable_with_rvalue_ref_to_cv_s3 const volatile&&,
                   s3 const volatile&&);

  operator s3 const volatile &() const volatile;
};
static_assert(
    std::swappable_with<swappable_with_rvalue_ref_to_cv_s3 const volatile&&,
                        s3 const volatile&&>);

namespace union_swap {
union adl_swappable {
  int x;
  double y;

  operator int() const;
};

void swap(adl_swappable&, adl_swappable&) noexcept;
void swap(adl_swappable&&, adl_swappable&&) noexcept;
void swap(adl_swappable&, int&) noexcept;
void swap(int&, adl_swappable&) noexcept;
} // namespace union_swap
static_assert(
    std::swappable_with<union_swap::adl_swappable, union_swap::adl_swappable>);
static_assert(std::swappable_with<union_swap::adl_swappable&,
                                  union_swap::adl_swappable&>);
static_assert(std::swappable_with<union_swap::adl_swappable&&,
                                  union_swap::adl_swappable&&>);
static_assert(std::swappable_with<union_swap::adl_swappable&, int&>);
} // namespace adl

namespace standard_types {
static_assert(
    check_swappable_with<std::array<int, 10>, std::array<int, 10> >());
static_assert(
    !check_swappable_with<std::array<int, 10>, std::array<double, 10> >());
static_assert(check_swappable_with<std::deque<int>, std::deque<int> >());
static_assert(!check_swappable_with<std::deque<int>, std::vector<int> >());
static_assert(
    check_swappable_with<std::forward_list<int>, std::forward_list<int> >());
static_assert(
    !check_swappable_with<std::forward_list<int>, std::vector<int> >());
static_assert(check_swappable_with<std::list<int>, std::list<int> >());
static_assert(!check_swappable_with<std::list<int>, std::vector<int> >());

static_assert(
    check_swappable_with<std::map<int, void*>, std::map<int, void*> >());
static_assert(!check_swappable_with<std::map<int, void*>, std::vector<int> >());
static_assert(check_swappable_with<std::optional<std::vector<int> >,
                                   std::optional<std::vector<int> > >());
static_assert(!check_swappable_with<std::optional<std::vector<int> >,
                                    std::vector<int> >());
static_assert(check_swappable_with<std::vector<int>, std::vector<int> >());
static_assert(!check_swappable_with<std::vector<int>, int>());
} // namespace standard_types

namespace types_with_purpose {
static_assert(!check_swappable_with<DeletedMoveCtor, DeletedMoveCtor>());
static_assert(!check_swappable_with<ImplicitlyDeletedMoveCtor,
                                    ImplicitlyDeletedMoveCtor>());
static_assert(!check_swappable_with<DeletedMoveAssign, DeletedMoveAssign>());
static_assert(!check_swappable_with<ImplicitlyDeletedMoveAssign,
                                    ImplicitlyDeletedMoveAssign>());
static_assert(!check_swappable_with<NonMovable, NonMovable>());
static_assert(
    !check_swappable_with<DerivedFromNonMovable, DerivedFromNonMovable>());
static_assert(!check_swappable_with<HasANonMovable, HasANonMovable>());
} // namespace types_with_purpose

int main(int, char**) { return 0; }
