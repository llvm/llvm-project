//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// struct owner_hash
// {
//     template<class T>
//         size_t operator()(shared_ptr<T> const&) const noexcept;
//     template<class T>
//         size_t operator()(weak_ptr<T> const&) const noexcept;
//
//     using is_transparent = unspecified;
// };

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>

#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::weak_ptr<int> w1(p1);

  std::owner_hash oh;

  std::same_as<std::size_t> decltype(auto) hash_p1 = oh(p1);
  std::same_as<std::size_t> decltype(auto) hash_w1 = oh(w1);
  static_assert(noexcept(oh(p1)));
  static_assert(noexcept(oh(w1)));

  assert(hash_p1 == p1.owner_hash());
  assert(hash_w1 == w1.owner_hash());
  assert(hash_p1 == oh(p2));
  assert(hash_p1 == hash_w1);

  using member_is_transparent [[maybe_unused]] = std::owner_hash::is_transparent;
  LIBCPP_STATIC_ASSERT(std::same_as<member_is_transparent, void>);

  return 0;
}
