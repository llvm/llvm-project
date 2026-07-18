//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <memory>

// struct owner_equal
// {
//     template<class T, class U>
//         bool operator()(shared_ptr<T> const&, shared_ptr<U> const&) const noexcept;
//     template<class T, class U>
//         bool operator()(shared_ptr<T> const&, weak_ptr<U> const&) const noexcept;
//     template<class T, class U>
//         bool operator()(weak_ptr<T> const&, shared_ptr<U> const&) const noexcept;
//     template<class T, class U>
//         bool operator()(weak_ptr<T> const&, weak_ptr<U> const&) const noexcept;
//
//     typedef unspecified is_transparent;
// };

#include <memory>
#include <cassert>
#include <type_traits>
#include <unordered_set>
#include "test_macros.h"

int main(int, char**) {
  const std::shared_ptr<int> p1(new int);
  const std::shared_ptr<int> p2 = p1;
  const std::shared_ptr<int> p3(new int);
  const std::weak_ptr<int> w1(p1);
  const std::weak_ptr<int> w3(p3);

  std::owner_equal oe;

  assert(oe(p1, p2));
  assert(oe(p1, w1));
  assert(oe(w1, p1));
  assert(oe(w1, w1));

  assert(!oe(p1, p3));
  assert(!oe(p1, w3));
  assert(!oe(w1, p3));
  assert(!oe(w1, w3));

  ASSERT_SAME_TYPE(decltype(oe(p1, p2)), bool);
  ASSERT_NOEXCEPT(oe(p1, p2));
  ASSERT_NOEXCEPT(oe(p1, w1));
  ASSERT_NOEXCEPT(oe(w1, p1));
  ASSERT_NOEXCEPT(oe(w1, w1));

  static_assert(std::is_same<std::owner_equal::is_transparent, void>::value, "");

  {
    std::unordered_set<std::weak_ptr<int>, std::owner_hash, std::owner_equal> s;
    s.insert(w1);
    assert(s.count(w1) == 1);
    assert(s.count(std::weak_ptr<int>(p2)) == 1);
    assert(s.count(w3) == 0);
  }

  return 0;
}
