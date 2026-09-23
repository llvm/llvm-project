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

// template<class U> bool owner_equal(weak_ptr<U> const& b) const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p3(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<int> w2(p1);
  const std::weak_ptr<int> w3(p3);
  const std::weak_ptr<void> empty1;
  const std::weak_ptr<long> empty2;

  std::same_as<bool> decltype(auto) result = w1.owner_equal(w2);
  assert(result);
  static_assert(noexcept(w1.owner_equal(w2)));

  assert(w2.owner_equal(w1));
  assert(!w1.owner_equal(w3));
  assert(!w3.owner_equal(w1));

  assert(empty1.owner_equal(empty2));
  assert(!w1.owner_equal(empty1));

  return 0;
}
