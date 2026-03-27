//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// optional(U&& u) noexcept(is_nothrow_constructible_v<T&, U>);
// optional(optional<U>& rhs) noexcept(is_nothrow_constructible_v<T&, U&>);
// optional(const optional<U>& rhs) noexcept(is_nothrow_constructible_v<T&, const U&>);
// optional(optional<U>&& rhs) noexcept(is_nothrow_constructible_v<T&, U>);
// optional(const optional<U>&& rhs) noexcept(is_nothrow_constructible_v<T&, const U>);

#include <optional>
#include <utility>

struct X {
  int i;

  X(int j) : i(j) {}
};

void test() {
  const std::optional<int> co(1);
  std::optional<int> o0(1);

  // expected-error-re@*:* 10 {{call to deleted constructor of 'std::optional<{{.*}}>'}}
  std::optional<const int&> o1{1};             // optional(U&&)
  std::optional<const int&> o2{o0};            // optional(optional<U>&)
  std::optional<const int&> o3{co};            // optional(const optional<U>&)
  std::optional<const int&> o4{std::move(o0)}; // optional(optional<U>&&&)
  std::optional<const int&> o5{std::move(co)}; // optional(optional<U>&&&)

  std::optional<const X&> o6{1};              // optional(U&&)
  std::optional<const X&> o7{o0};             // optional(optional<U>&)
  std::optional<const X&> o8(co);             // optional(const optional<U>&)
  std::optional<const X&> o9{std::move(o0)};  // optional(optional<U>&&)
  std::optional<const X&> o10{std::move(co)}; // optional(const optional<U>&&)
}
