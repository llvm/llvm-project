//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// void swap(hazard_pointer& a, hazard_pointer& b) noexcept;   Equivalent to a.swap(b).

#include <hazard_pointer>
#include <cassert>

#include "test_macros.h"

template <class T>
concept IsNoThrowFreeSwappable = requires(T& t) {
  { swap(t, t) } noexcept;
};
static_assert(IsNoThrowFreeSwappable<std::hazard_pointer>);

int main(int, char**) {
  std::hazard_pointer a = std::make_hazard_pointer(), b;
  swap(a, b); // ADL
  assert(a.empty());
  assert(!b.empty());
  std::swap(a, b);
  assert(!a.empty());
  assert(b.empty());
  return 0;
}
