//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// <flat_set>

// Test that the invariants are restored when inserting an element in the middle of the
// underlying container throws from the element's assignment operator. An insertion in the
// middle of a vector or deque has no "no effects" guarantee in that case, even if the
// element type is Cpp17CopyInsertable.

#include <cassert>
#include <compare>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "../helpers.h"

struct ThrowingAssign {
  int value;
  bool* throw_on_assign;

  ThrowingAssign(int v, bool& t) : value(v), throw_on_assign(&t) {}
  ThrowingAssign(const ThrowingAssign&)     = default;
  ThrowingAssign(ThrowingAssign&&) noexcept = default;
  ThrowingAssign& operator=(const ThrowingAssign& other) {
    if (*throw_on_assign)
      throw 42;
    value = other.value;
    return *this;
  }
  ThrowingAssign& operator=(ThrowingAssign&& other) {
    if (*throw_on_assign)
      throw 42;
    value = other.value;
    return *this;
  }

  friend bool operator==(const ThrowingAssign& x, const ThrowingAssign& y) { return x.value == y.value; }
  friend auto operator<=>(const ThrowingAssign& x, const ThrowingAssign& y) { return x.value <=> y.value; }
};

template <class KeyContainer, class F>
void test_one(F insert_function) {
  using M = std::flat_set<ThrowingAssign, std::less<ThrowingAssign>, KeyContainer>;

  bool throw_on_assign = false;
  KeyContainer keys;
  // Spare capacity makes vector insert in place instead of reallocating.
  if constexpr (requires { keys.reserve(8); })
    keys.reserve(8);
  for (int v : {1, 3, 5})
    keys.emplace_back(v, throw_on_assign);
  M m(std::sorted_unique, std::move(keys));

  throw_on_assign = true;
  try {
    insert_function(m, throw_on_assign);
    assert(false);
  } catch (int) {
  }
  throw_on_assign = false;
  check_invariant(m);
}

template <class KeyContainer>
void test() {
  test_one<KeyContainer>([](auto& m, bool& t) { m.emplace(2, t); });
  test_one<KeyContainer>([](auto& m, bool& t) { m.emplace_hint(m.begin() + 1, 2, t); });
  test_one<KeyContainer>([](auto& m, bool& t) { m.insert(ThrowingAssign(2, t)); });
  test_one<KeyContainer>([](auto& m, bool& t) { m.insert(m.begin() + 1, ThrowingAssign(2, t)); });
}

int main(int, char**) {
  test<std::vector<ThrowingAssign>>();
  test<std::deque<ThrowingAssign>>();

  return 0;
}
