//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class... Args>
// concept equivalence_relation;

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
requires std::relation<F, T, U>
constexpr bool check_equivalence_relation_subsumes_relation() {
  return false;
}

template<class F, class T, class U>
requires std::equivalence_relation<F, T, U> && true
constexpr bool check_equivalence_relation_subsumes_relation() {
  return true;
}
// clang-format on

static_assert(check_equivalence_relation_subsumes_relation<int (*)(int, int),
                                                           int, int>());
static_assert(check_equivalence_relation_subsumes_relation<int (*)(int, double),
                                                           int, double>());
static_assert(check_equivalence_relation_subsumes_relation<R, S1, S1>());
static_assert(check_equivalence_relation_subsumes_relation<R, S1, S2>());

// clang-format off
template<class F, class T, class U>
requires std::relation<F, T, U> && true
constexpr bool check_relation_subsumes_equivalence_relation() {
  return true;
}

template<class F, class T, class U>
requires std::equivalence_relation<F, T, U>
constexpr bool check_relation_subsumes_equivalence_relation() {
  return false;
}
// clang-format on

static_assert(check_relation_subsumes_equivalence_relation<int (*)(int, int),
                                                           int, int>());
static_assert(check_relation_subsumes_equivalence_relation<int (*)(int, double),
                                                           int, double>());
static_assert(check_relation_subsumes_equivalence_relation<R, S1, S1>());
static_assert(check_relation_subsumes_equivalence_relation<R, S1, S2>());

// clang-format off
template<class F, class T, class U>
requires std::equivalence_relation<F, T, T> && std::equivalence_relation<F, U, U>
constexpr bool check_equivalence_relation_subsumes_itself() {
  return false;
}

template<class F, class T, class U>
requires std::equivalence_relation<F, T, U>
constexpr bool check_equivalence_relation_subsumes_itself() {
  return true;
}
// clang-format on

static_assert(
    check_equivalence_relation_subsumes_itself<int (*)(int, int), int, int>());
static_assert(check_equivalence_relation_subsumes_itself<R, S1, S1>());
