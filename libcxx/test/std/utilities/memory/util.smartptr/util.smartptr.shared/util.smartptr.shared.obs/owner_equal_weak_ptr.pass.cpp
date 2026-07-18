//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <memory>

// shared_ptr

// template<class U> bool owner_equal(weak_ptr<U> const& b) const noexcept;

#include <memory>
#include <cassert>
#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::shared_ptr<int> p3(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<int> w3(p3);
  const std::shared_ptr<void> empty_sp;
  const std::weak_ptr<long> empty_wp;

  assert(p1.owner_equal(w1));
  assert(p2.owner_equal(w1));
  assert(!p1.owner_equal(w3));
  assert(!p3.owner_equal(w1));

  assert(empty_sp.owner_equal(empty_wp));

  ASSERT_SAME_TYPE(decltype(p1.owner_equal(w1)), bool);
  ASSERT_NOEXCEPT(p1.owner_equal(w1));

  return 0;
}
