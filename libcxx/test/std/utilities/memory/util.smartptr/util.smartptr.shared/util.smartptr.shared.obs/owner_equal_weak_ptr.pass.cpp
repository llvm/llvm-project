//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// shared_ptr

// template<class U> bool owner_equal(weak_ptr<U> const& b) const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::shared_ptr<int> p3(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<int> w3(p3);
  const std::shared_ptr<void> empty_sp;
  const std::weak_ptr<long> empty_wp;

  std::same_as<bool> decltype(auto) result = p1.owner_equal(w1);
  assert(result);
  static_assert(noexcept(p1.owner_equal(w1)));

  assert(p2.owner_equal(w1));
  assert(!p1.owner_equal(w3));
  assert(!p3.owner_equal(w1));

  assert(empty_sp.owner_equal(empty_wp));

  return 0;
}
