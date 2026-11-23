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

// size_t owner_hash() const noexcept;

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>

#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<int> w2(p1);

  std::same_as<std::size_t> decltype(auto) hash1 = w1.owner_hash();
  static_assert(noexcept(w1.owner_hash()));

  assert(hash1 == w2.owner_hash());
  assert(hash1 == p1.owner_hash());

  return 0;
}
