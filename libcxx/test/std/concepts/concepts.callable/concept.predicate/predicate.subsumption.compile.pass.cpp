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

constexpr bool check_subsumption(std::regular_invocable auto) {
  return false;
}

// clang-format off
template<class F>
requires std::predicate<F> && true
constexpr bool check_subsumption(F)
{
  return true;
}
// clang-format on

static_assert(!check_subsumption([] {}));
static_assert(check_subsumption([] { return true; }));
