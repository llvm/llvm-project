//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types>
// constexpr bool
// operator==(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator!=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>=(variant<Types...> const&, variant<Types...> const&) noexcept;

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"


struct MyBoolExplicit {
  bool value;
  constexpr explicit MyBoolExplicit(bool v) : value(v) {}
  constexpr explicit operator bool() const noexcept { return value; }
};

struct ComparesToMyBoolExplicit {
  int value = 0;
};
inline constexpr MyBoolExplicit operator==(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value == RHS.value);
}
inline constexpr MyBoolExplicit operator!=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value != RHS.value);
}
inline constexpr MyBoolExplicit operator<(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value < RHS.value);
}
inline constexpr MyBoolExplicit operator<=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value <= RHS.value);
}
inline constexpr MyBoolExplicit operator>(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value > RHS.value);
}
inline constexpr MyBoolExplicit operator>=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept {
  return MyBoolExplicit(LHS.value >= RHS.value);
}


int main(int, char**) {
  using V = std::variant<int, ComparesToMyBoolExplicit>;
  V v1(42);
  V v2(101);
  // expected-error-re@variant:* 6 {{{{(static_assert|static assertion)}} failed{{.*}}the relational operator does not return a type which is implicitly convertible to bool}}
  // expected-error@variant:* 6 {{no viable conversion}}
  (void)(v1 == v2); // expected-note {{here}}
  (void)(v1 != v2); // expected-note {{here}}
  (void)(v1 < v2); // expected-note {{here}}
  (void)(v1 <= v2); // expected-note {{here}}
  (void)(v1 > v2); // expected-note {{here}}
  (void)(v1 >= v2); // expected-note {{here}}

  return 0;
}
