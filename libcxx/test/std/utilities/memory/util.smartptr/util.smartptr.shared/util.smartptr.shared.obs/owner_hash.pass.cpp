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

// size_t owner_hash() const noexcept;

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>

#include "test_macros.h"

struct Pair {
  int a;
  int b;
};

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::weak_ptr<int> w1(p1);

  std::same_as<std::size_t> decltype(auto) hash1 = p1.owner_hash();
  static_assert(noexcept(p1.owner_hash()));

  assert(hash1 == p2.owner_hash());
  assert(hash1 == w1.owner_hash());

  const std::shared_ptr<Pair> sp(new Pair{1, 2});
  const std::shared_ptr<int> alias(sp, &sp->b);
  assert(static_cast<void*>(sp.get()) != static_cast<void*>(alias.get()));
  assert(sp.owner_hash() == alias.owner_hash());

  return 0;
}
