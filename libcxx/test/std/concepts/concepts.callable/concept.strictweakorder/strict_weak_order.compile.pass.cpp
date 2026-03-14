//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class... Args>
// concept strict_weak_order;

#include <concepts>

static_assert(std::strict_weak_order<bool(int, int), int, int>);
static_assert(std::strict_weak_order<bool(int, int), double, double>);
static_assert(std::strict_weak_order<bool(int, double), double, double>);

static_assert(!std::strict_weak_order<bool (*)(), int, double>);
static_assert(!std::strict_weak_order<bool (*)(int), int, double>);
static_assert(!std::strict_weak_order<bool (*)(double), int, double>);

static_assert(!std::strict_weak_order<bool(double, double*), double, double*>);
static_assert(!std::strict_weak_order<bool(int&, int&), double&, double&>);

struct S1 {};
static_assert(std::strict_weak_order<bool (S1::*)(S1*), S1*, S1*>);
static_assert(std::strict_weak_order<bool (S1::*)(S1&), S1&, S1&>);

struct S2 {};

struct P1 {
  bool operator()(S1, S1) const;
};
static_assert(std::strict_weak_order<P1, S1, S1>);

struct P2 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
};
static_assert(!std::strict_weak_order<P2, S1, S2>);

struct P3 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
};
static_assert(!std::strict_weak_order<P3, S1, S2>);

struct P4 {
  bool operator()(S1, S1) const;
  bool operator()(S1, S2) const;
  bool operator()(S2, S1) const;
  bool operator()(S2, S2) const;
};
static_assert(std::strict_weak_order<P4, S1, S2>);
