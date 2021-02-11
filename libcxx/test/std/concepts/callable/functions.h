//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CALLABLE_FUNCTIONS_H
#define CALLABLE_FUNCTIONS_H

namespace RegularInvocable {
struct A {
  int I = 13;
  constexpr int F() const noexcept { return 42; }
  constexpr int G(int X) { return 2 * X + 1; }
  constexpr int H(int J) && { return I * J; }
};

constexpr int F() noexcept { return 13; }
constexpr int G(int I) { return 2 * I + 1; }
} // namespace RegularInvocable

namespace Predicate {
struct L2rSorted {
  template <class T>
  constexpr bool operator()(T const& A, T const& B, T const& C) const noexcept {
    return A <= B && B <= C;
  }
};

struct NotAPredicate {
  void operator()() const noexcept {}
};
} // namespace Predicate

namespace Relation {
int Greater(int X, int Y) noexcept { return X > Y; }
} // namespace Relation

#endif // CALLABLE_FUNCTIONS_H
