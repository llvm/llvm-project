//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// template<auto f> constexpr unspecified not_fn() noexcept;
// Mandates: If is_pointer_v<F> || is_member_pointer_v<F> is true, then f != nullptr is true.

#include <functional>

struct X {};

void test() {
  auto not_fn1 = std::not_fn<static_cast<bool (*)()>(nullptr)>();
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': f cannot be equal to nullptr}}

  auto not_fn2 = std::not_fn<static_cast<bool X::*>(nullptr)>();
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': f cannot be equal to nullptr}}

  auto not_fn3 = std::not_fn<static_cast<bool (X::*)()>(nullptr)>();
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': f cannot be equal to nullptr}}
}
