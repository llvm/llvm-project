//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// weak_ptr

// template<class U> bool owner_equal(shared_ptr<U> const& b) const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p3(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<void> empty;

  std::same_as<bool> decltype(auto) result = w1.owner_equal(p1);
  assert(result);
  static_assert(noexcept(w1.owner_equal(p1)));

  assert(!w1.owner_equal(p3));

  const std::shared_ptr<long> empty_sp;
  assert(empty.owner_equal(empty_sp));

  return 0;
}
