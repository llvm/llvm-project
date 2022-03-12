//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class... Args>
// concept relation;

#include <concepts>

// clang-format off
template<class F, class T, class U>
requires std::predicate<F, T, T> && std::predicate<F, T, U> &&
         std::predicate<F, U, T> && std::predicate<F, U, U>
constexpr bool check_subsumption() { return false; }

template<class F, class T, class U>
requires std::relation<F, T, U> && true
constexpr bool check_subsumption() { return true; }
// clang-format on

static_assert(check_subsumption<int (*)(int, double), int, double>());

struct S1 {};
struct S2 {};

struct R {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
  bool operator()(S2, S2) const;
};
static_assert(check_subsumption<R, S1, S2>());

int main(int, char**) { return 0; }
