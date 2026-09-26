//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class... Args>
//   constexpr optional<reference> try_emplace_back(Args&&... args);
// constexpr optional<reference> try_push_back(const T& x);
// constexpr optional<reference> try_push_back(T&& x);

#include <cassert>
#include <concepts>
#include <inplace_vector>
#include <optional>
#include <utility>

#include "../common.h"
#include "MoveOnly.h"
#include "test_macros.h"

struct A {
  int i;
  int j;

  constexpr A(int ii, int jj) : i(ii), j(jj) {}
};

constexpr bool test() {
  {
    std::inplace_vector<int, 2> c;
    int value = 1;

    std::same_as<std::optional<int&>> decltype(auto) r1 = c.try_push_back(value);
    assert(r1);
    assert(&*r1 == &c.back());
    assert_inplace_vector_equal(c, {1});

    std::same_as<std::optional<int&>> decltype(auto) r2 = c.try_push_back(2);
    assert(r2);
    assert(&*r2 == &c.back());
    assert_inplace_vector_equal(c, {1, 2});

    auto r3 = c.try_push_back(3);
    assert(!r3);
    assert_inplace_vector_equal(c, {1, 2});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<A, 1> c;
    std::same_as<std::optional<A&>> decltype(auto) r = c.try_emplace_back(1, 2);
    assert(r);
    assert(&*r == &c.back());
    assert(r->i == 1);
    assert(r->j == 2);

    auto r2 = c.try_emplace_back(3, 4);
    assert(!r2);
    assert(c.size() == 1);
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<MoveOnly, 1> c;
    auto r = c.try_push_back(MoveOnly(1));
    assert(r);
    assert(r->get() == 1);
    auto r2 = c.try_push_back(MoveOnly(2));
    assert(!r2);
    assert(c[0].get() == 1);
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    // when there is no effect, the argument is not moved from
    std::inplace_vector<MoveOnly, 1> c;
    (void)c.try_push_back(MoveOnly(1));
    MoveOnly m(2);
    auto r = c.try_push_back(std::move(m));
    assert(!r);
    assert(m.get() == 2);
    assert(c[0].get() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // an exception thrown by T's constructor propagates and there are no effects
    ThrowingValue::reset();
    {
      std::inplace_vector<ThrowingValue, 4> c;
      (void)c.try_emplace_back(1);
      assert(c.size() == 1);

      ThrowingValue::throw_after = 0;
      try {
        (void)c.try_emplace_back(2);
        assert(false);
      } catch (int) {
      }
      ThrowingValue::throw_after = -1;
      assert(c.size() == 1);
      assert(c[0].value == 1);
    }
    assert(ThrowingValue::alive == 0);
    ThrowingValue::reset();
  }
#endif

  return 0;
}
