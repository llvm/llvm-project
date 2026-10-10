//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// <flat_map>

// Test that the invariants are restored when inserting an element in the middle of the
// underlying containers throws from the element's assignment operator or, during
// reallocation, from its copy constructor. Neither has a "no effects" guarantee for an
// insertion in the middle of a vector or deque, even if the element type is
// Cpp17CopyInsertable.

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "../helpers.h"

struct ThrowingAssign {
  static inline bool throw_on_assign = false;

  int value;

  ThrowingAssign(int v) : value(v) {}
  ThrowingAssign(const ThrowingAssign&)     = default;
  ThrowingAssign(ThrowingAssign&&) noexcept = default;
  ThrowingAssign& operator=(const ThrowingAssign& other) {
    if (throw_on_assign)
      throw 42;
    value = other.value;
    return *this;
  }
  ThrowingAssign& operator=(ThrowingAssign&& other) {
    if (throw_on_assign)
      throw 42;
    value = other.value;
    return *this;
  }

  friend bool operator==(const ThrowingAssign&, const ThrowingAssign&)  = default;
  friend auto operator<=>(const ThrowingAssign&, const ThrowingAssign&) = default;
};

struct ThrowingCopy {
  static inline int copies_until_throw = -1; // -1: never throw

  int value;

  ThrowingCopy(int v) : value(v) {}
  ThrowingCopy(const ThrowingCopy& other) : value(other.value) {
    if (copies_until_throw >= 0 && copies_until_throw-- == 0)
      throw 42;
  }
  ThrowingCopy(ThrowingCopy&& other) : ThrowingCopy(other) {}
  ThrowingCopy& operator=(const ThrowingCopy&) noexcept = default;
  ThrowingCopy& operator=(ThrowingCopy&&) noexcept      = default;

  friend bool operator==(const ThrowingCopy&, const ThrowingCopy&)  = default;
  friend auto operator<=>(const ThrowingCopy&, const ThrowingCopy&) = default;
};

// n >= 0: assignments throw, and the n-th copy (counting from 0) throws. n < 0: nothing throws.
void set_throwing(int n) {
  ThrowingAssign::throw_on_assign  = n >= 0;
  ThrowingCopy::copies_until_throw = n;
}

template <class Container>
Container make_container(std::initializer_list<int> il) {
  Container c;
  for (int v : il)
    c.emplace_back(v);
  if constexpr (requires { c.reserve(8); }) {
    // ThrowingAssign: spare capacity makes vector insert in place.
    // ThrowingCopy: no spare capacity makes vector reallocate.
    if constexpr (std::is_same_v<typename Container::value_type, ThrowingCopy>)
      c.shrink_to_fit();
    else
      c.reserve(8);
  }
  return c;
}

template <class KeyContainer, class ValueContainer, class F>
void test_one(F insert_function) {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  for (int n = 0; n < 6; ++n) {
    M m(std::sorted_unique, make_container<KeyContainer>({1, 3, 5}), make_container<ValueContainer>({10, 30, 50}));
    set_throwing(n);
    try {
      insert_function(m);
    } catch (int) {
    }
    set_throwing(-1);
    check_invariant(m);
  }
}

template <class KeyContainer, class ValueContainer>
void test() {
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.emplace(2, 20); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.emplace_hint(m.begin() + 1, 2, 20); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.try_emplace(2, 20); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.try_emplace(m.begin() + 1, 2, 20); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.insert({2, 20}); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.insert(m.begin() + 1, {2, 20}); });
  test_one<KeyContainer, ValueContainer>([](auto& m) { m.insert_or_assign(2, 20); });
}

int main(int, char**) {
  // Throw while inserting the key.
  test<std::vector<ThrowingAssign>, std::vector<int>>();
  test<std::deque<ThrowingAssign>, std::vector<int>>();
  // Throw while inserting the mapped value.
  test<std::vector<int>, std::vector<ThrowingAssign>>();
  test<std::vector<int>, std::deque<ThrowingAssign>>();
  // Throw while reallocating.
  test<std::vector<ThrowingCopy>, std::vector<int>>();
  test<std::vector<int>, std::vector<ThrowingCopy>>();

  return 0;
}
