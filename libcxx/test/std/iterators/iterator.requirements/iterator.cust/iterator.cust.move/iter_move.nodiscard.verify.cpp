//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// Test the [[nodiscard]] extension in libc++.

// template<class I>
// unspecified iter_move;

#include <iterator>

struct WithADL {
  WithADL() = default;
  constexpr decltype(auto) operator*() const noexcept;
  constexpr WithADL& operator++() noexcept;
  constexpr void operator++(int) noexcept;
  constexpr bool operator==(WithADL const&) const noexcept;
  friend constexpr auto iter_move(WithADL&) { return 0; }
};

int main(int, char**) {
  int* noADL = nullptr;
  std::ranges::iter_move(noADL); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  WithADL adl;
  std::ranges::iter_move(adl); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  return 0;
}
