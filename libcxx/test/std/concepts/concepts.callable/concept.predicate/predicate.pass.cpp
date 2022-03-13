//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class F, class... Args>
// concept predicate;

#include <concepts>
#include <type_traits>

static_assert(std::predicate<bool()>);
static_assert(std::predicate<bool (*)()>);
static_assert(std::predicate<bool (&)()>);

static_assert(!std::predicate<void()>);
static_assert(!std::predicate<void (*)()>);
static_assert(!std::predicate<void (&)()>);

struct S {};

static_assert(!std::predicate<S(int), int>);
static_assert(!std::predicate<S(double), double>);
static_assert(std::predicate<int S::*, S*>);
static_assert(std::predicate<int (S::*)(), S*>);
static_assert(std::predicate<int (S::*)(), S&>);
static_assert(!std::predicate<void (S::*)(), S*>);
static_assert(!std::predicate<void (S::*)(), S&>);

static_assert(!std::predicate<bool(S)>);
static_assert(!std::predicate<bool(S&), S>);
static_assert(!std::predicate<bool(S&), S const&>);
static_assert(std::predicate<bool(S&), S&>);

struct Predicate {
  bool operator()(int, double, char);
};
static_assert(std::predicate<Predicate, int, double, char>);
static_assert(std::predicate<Predicate&, int, double, char>);
static_assert(!std::predicate<const Predicate, int, double, char>);
static_assert(!std::predicate<const Predicate&, int, double, char>);

constexpr bool check_lambda(auto) { return false; }

constexpr bool check_lambda(std::predicate auto) { return true; }

static_assert(check_lambda([] { return std::true_type(); }));
static_assert(check_lambda([]() -> int* { return nullptr; }));

struct boolean {
  operator bool() const noexcept;
};
static_assert(check_lambda([] { return boolean(); }));

struct explicit_bool {
  explicit operator bool() const noexcept;
};
static_assert(!check_lambda([] { return explicit_bool(); }));

int main(int, char**) { return 0; }
