//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <memory>

// struct owner_hash
// {
//     template<class T>
//         size_t operator()(shared_ptr<T> const&) const noexcept;
//     template<class T>
//         size_t operator()(weak_ptr<T> const&) const noexcept;
//
//     typedef unspecified is_transparent;
// };

#include <memory>
#include <cassert>
#include <type_traits>
#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::weak_ptr<int> w1(p1);

  std::owner_hash oh;

  assert(oh(p1) == p1.owner_hash());
  assert(oh(w1) == w1.owner_hash());

  assert(oh(p1) == oh(p2));
  assert(oh(p1) == oh(w1));

  ASSERT_SAME_TYPE(decltype(oh(p1)), std::size_t);
  ASSERT_SAME_TYPE(decltype(oh(w1)), std::size_t);
  ASSERT_NOEXCEPT(oh(p1));
  ASSERT_NOEXCEPT(oh(w1));

  static_assert(std::is_same<std::owner_hash::is_transparent, void>::value, "");

  return 0;
}
