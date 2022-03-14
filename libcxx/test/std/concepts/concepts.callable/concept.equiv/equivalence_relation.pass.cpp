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

static_assert(std::equivalence_relation<bool(int, int), int, int>);
static_assert(std::equivalence_relation<bool(int, int), double, double>);
static_assert(std::equivalence_relation<bool(int, double), double, double>);

static_assert(!std::equivalence_relation<bool (*)(), int, double>);
static_assert(!std::equivalence_relation<bool (*)(int), int, double>);
static_assert(!std::equivalence_relation<bool (*)(double), int, double>);

static_assert(
    !std::equivalence_relation<bool(double, double*), double, double*>);
static_assert(!std::equivalence_relation<bool(int&, int&), double&, double&>);

struct S1 {};
static_assert(std::relation<bool (S1::*)(S1*), S1*, S1*>);
static_assert(std::relation<bool (S1::*)(S1&), S1&, S1&>);

struct S2 {};

struct P1 {
  bool operator()(S1, S1) const;
};
static_assert(std::equivalence_relation<P1, S1, S1>);

struct P2 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
};
static_assert(!std::equivalence_relation<P2, S1, S2>);

struct P3 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
};
static_assert(!std::equivalence_relation<P3, S1, S2>);

struct P4 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
  bool operator()(S2, S2) const;
};
static_assert(std::equivalence_relation<P4, S1, S2>);

int main(int, char**) { return 0; }
