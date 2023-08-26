//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;
//     Mandates: one of pointer_traits<Ptr>::to_address() or Ptr::operator->()
//     is present.

#include <memory>

struct NotPtr {};

void test() {
  (void)std::to_address(NotPtr()); // expected-error@*:* {{no matching function for call to 'to_address'}}
}
