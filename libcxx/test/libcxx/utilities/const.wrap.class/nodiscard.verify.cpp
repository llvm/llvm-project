//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Check that functions are marked [[nodiscard]]

#include <utility>

constexpr int i = 5;

struct Ops {
  friend constexpr Ops operator&&(Ops, Ops) { return {}; }
  friend constexpr Ops operator||(Ops, Ops) { return {}; }
  friend constexpr Ops operator->*(Ops, Ops) { return {}; }

  constexpr Ops operator++() const { return {}; }
  constexpr Ops operator++(int) const { return {}; }
  constexpr Ops operator--() const { return {}; }
  constexpr Ops operator--(int) const { return {}; }
  constexpr Ops operator+=(Ops) const { return {}; }
  constexpr Ops operator-=(Ops) const { return {}; }
  constexpr Ops operator*=(Ops) const { return {}; }
  constexpr Ops operator/=(Ops) const { return {}; }
  constexpr Ops operator%=(Ops) const { return {}; }
  constexpr Ops operator&=(Ops) const { return {}; }
  constexpr Ops operator|=(Ops) const { return {}; }
  constexpr Ops operator^=(Ops) const { return {}; }
  constexpr Ops operator<<=(Ops) const { return {}; }
  constexpr Ops operator>>=(Ops) const { return {}; }

  constexpr Ops operator=(auto) const { return {}; }
};

void test() {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  +std::cw<5>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  -std::cw<5>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ~std::cw<5>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  !std::cw<5>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  &std::cw<5>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::cw<&i>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> + std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> - std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1>* std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> / std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> % std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> << std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> >> std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1>& std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> | std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<1> ^ std::cw<2>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}>&& std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> || std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}>->*std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ++std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> --;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  --std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> --;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> += std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> -= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> *= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> /= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> %= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> &= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> |= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> ^= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> <<= std::cw<Ops{}>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cw<Ops{}> >>= std::cw<Ops{}>;

  std::constant_wrapper<Ops{}> a;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a = std::cw<5>;
}
