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
// concept equivalence_relation;

#include <concepts>

// clang-format off
template<class F, class T, class U>
requires std::relation<F, T, U>
[[nodiscard]] constexpr bool check_subsumption() { return false; }

template<class F, class T>
requires std::equivalence_relation<F, T, T> && true
[[nodiscard]] constexpr bool check_subsumption() { return false; }

template<class F, class T, class U>
requires std::equivalence_relation<F, T, U> && true
[[nodiscard]] constexpr bool check_subsumption() { return true; }
// clang-format on

static_assert(check_subsumption<int (*)(int, int), int, int>());
static_assert(check_subsumption<int (*)(int, double), int, double>());

struct S1 {};
struct S2 {};

struct R {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
  bool operator()(S2, S2) const;
};
static_assert(check_subsumption<R, S1, S1>());
static_assert(check_subsumption<R, S1, S2>());

// clang-format off
template<class F, class T, class U>
requires std::relation<F, T, U> && true
[[nodiscard]] constexpr bool check_reverse_subsumption() { return true; }

template<class F, class T, class U>
requires std::equivalence_relation<F, T, U>
[[nodiscard]] constexpr bool check_no_subsumption() { return false; }
// clang-format on

static_assert(check_reverse_subsumption<int (*)(int, int), int, int>());
static_assert(check_reverse_subsumption<int (*)(int, double), int, double>());
static_assert(check_reverse_subsumption<R, S1, S1>());
static_assert(check_reverse_subsumption<R, S1, S2>());

int main(int, char**) { return 0; }
