//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class F, class... Args>
// concept relation;

#include <concepts>

struct S1 {};
struct S2 {};

struct R {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
  bool operator()(S2, S2) const;
};

// clang-format off
template<class F, class T, class U>
requires std::predicate<F, T, T> && std::predicate<F, T, U> &&
         std::predicate<F, U, T> && std::predicate<F, U, U>
[[nodiscard]] constexpr bool check_relation_subsumes_predicate() {
  return false;
}

template<class F, class T, class U>
requires std::relation<F, T, U> && true
[[nodiscard]] constexpr bool check_relation_subsumes_predicate() {
  return true;
}
// clang-format on

static_assert(
    check_relation_subsumes_predicate<int (*)(int, double), int, int>());
static_assert(
    check_relation_subsumes_predicate<int (*)(int, double), int, double>());
static_assert(check_relation_subsumes_predicate<R, S1, S1>());
static_assert(check_relation_subsumes_predicate<R, S1, S2>());

// clang-format off
template<class F, class T, class U>
requires std::relation<F, T, T> && std::relation<F, U, U>
[[nodiscard]] constexpr bool check_relation_subsumes_itself() {
  return false;
}

template<class F, class T, class U>
requires std::relation<F, T, U>
[[nodiscard]] constexpr bool check_relation_subsumes_itself() {
  return true;
}
// clang-format on

static_assert(check_relation_subsumes_itself<int (*)(int, double), int, int>());
static_assert(check_relation_subsumes_itself<R, S1, S1>());
