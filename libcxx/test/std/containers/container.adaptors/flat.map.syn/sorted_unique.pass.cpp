//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// struct sorted_unique_t { explicit sorted_unique_t() = default; };
// inline constexpr sorted_unique_t sorted_unique{};

#include <cassert>
#include <concepts>
#include <flat_map>
#include <type_traits>

template <class T>
void implicit_test(T) {}

template <class T>
concept HasImplicitDefaultCtor = requires { implicit_test<T>({}); };

static_assert(std::is_default_constructible_v<std::sorted_unique_t>);
static_assert(std::is_trivially_default_constructible_v<std::sorted_unique_t>);
static_assert(!HasImplicitDefaultCtor<std::sorted_unique_t>);

constexpr bool test() {
  { [[maybe_unused]] std::sorted_unique_t s; }
  { [[maybe_unused]] std::same_as<const std::sorted_unique_t&> decltype(auto) s = (std::sorted_unique); }
  { [[maybe_unused]] std::same_as<const std::sorted_unique_t> decltype(auto) copy = std::sorted_unique; }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
