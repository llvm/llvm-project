//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// struct nostopstate_t {
//   explicit nostopstate_t() = default;
// };
//
// inline constexpr nostopstate_t nostopstate{};

#include <concepts>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_trivially_default_constructible_v<std::nostopstate_t>);

struct Empty {};
static_assert(sizeof(Empty) == sizeof(std::nostopstate_t));

template <class T>
void conversionTest(T);

template <class T>
concept ImplicitlyDefaultConstructible = requires { conversionTest<T>({}); };
static_assert(!ImplicitlyDefaultConstructible<std::nostopstate_t>);

int main(int, char**) {
  [[maybe_unused]] std::same_as<std::nostopstate_t> auto x = std::nostopstate;
  [[maybe_unused]] auto y                                  = std::nostopstate_t{};

  return 0;
}
