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
#include <compare>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "../helpers.h"

enum class ThrowMode { Never, OnAssign, OnNthCopy };

struct ThrowState {
  ThrowMode mode         = ThrowMode::Never;
  int copies_until_throw = 0; // Only used with ThrowMode::OnNthCopy.
};

struct ThrowingAssign {
  static constexpr ThrowMode throw_mode = ThrowMode::OnAssign;

  int value;
  ThrowState* state;

  ThrowingAssign(int v, ThrowState& s) : value(v), state(&s) {}
  ThrowingAssign(const ThrowingAssign&)     = default;
  ThrowingAssign(ThrowingAssign&&) noexcept = default;
  ThrowingAssign& operator=(const ThrowingAssign& other) {
    if (state->mode == ThrowMode::OnAssign)
      throw 42;
    value = other.value;
    return *this;
  }
  ThrowingAssign& operator=(ThrowingAssign&& other) {
    if (state->mode == ThrowMode::OnAssign)
      throw 42;
    value = other.value;
    return *this;
  }

  friend bool operator==(const ThrowingAssign& x, const ThrowingAssign& y) { return x.value == y.value; }
  friend auto operator<=>(const ThrowingAssign& x, const ThrowingAssign& y) { return x.value <=> y.value; }
};

struct ThrowingCopy {
  static constexpr ThrowMode throw_mode = ThrowMode::OnNthCopy;

  int value;
  ThrowState* state;

  ThrowingCopy(int v, ThrowState& s) : value(v), state(&s) {}
  ThrowingCopy(const ThrowingCopy& other) : value(other.value), state(other.state) {
    if (state->mode == ThrowMode::OnNthCopy && state->copies_until_throw-- == 0)
      throw 42;
  }
  ThrowingCopy(ThrowingCopy&& other) : ThrowingCopy(other) {}
  ThrowingCopy& operator=(const ThrowingCopy&) noexcept = default;
  ThrowingCopy& operator=(ThrowingCopy&&) noexcept      = default;

  friend bool operator==(const ThrowingCopy& x, const ThrowingCopy& y) { return x.value == y.value; }
  friend auto operator<=>(const ThrowingCopy& x, const ThrowingCopy& y) { return x.value <=> y.value; }
};

template <class T>
T make_element(int v, ThrowState& state) {
  if constexpr (std::is_same_v<T, int>)
    return v;
  else
    return T(v, state);
}

template <class Container>
Container make_container(std::initializer_list<int> il, ThrowState& state) {
  using T = typename Container::value_type;
  Container c;
  for (int v : il)
    c.push_back(make_element<T>(v, state));
  if constexpr (requires { c.reserve(8); }) {
    // ThrowingAssign: spare capacity makes vector insert in place.
    // ThrowingCopy: no spare capacity makes vector reallocate.
    if constexpr (std::is_same_v<T, ThrowingCopy>)
      c.shrink_to_fit();
    else
      c.reserve(8);
  }
  return c;
}

template <class KeyContainer, class ValueContainer, class F>
void test_one(F insert_function) {
  using Key      = typename KeyContainer::value_type;
  using Value    = typename ValueContainer::value_type;
  using M        = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using Throwing = std::conditional_t<std::is_same_v<Key, int>, Value, Key>;

  // For ThrowingCopy, throw on each of the first few copies in turn.
  for (int n = 0; n < 6; ++n) {
    ThrowState state;
    M m(std::sorted_unique,
        make_container<KeyContainer>({1, 3, 5}, state),
        make_container<ValueContainer>({10, 30, 50}, state));
    Key key     = make_element<Key>(2, state);
    Value value = make_element<Value>(20, state);

    state.mode               = Throwing::throw_mode;
    state.copies_until_throw = n;
    try {
      insert_function(m, key, value);
    } catch (int) {
    }
    state.mode = ThrowMode::Never;
    check_invariant(m);
  }
}

template <class KeyContainer, class ValueContainer>
void test() {
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) { m.emplace(k, v); });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) {
    m.emplace_hint(m.begin() + 1, k, v);
  });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) { m.try_emplace(k, v); });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) {
    m.try_emplace(m.begin() + 1, k, v);
  });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) { m.insert({k, v}); });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) {
    m.insert(m.begin() + 1, {k, v});
  });
  test_one<KeyContainer, ValueContainer>([](auto& m, const auto& k, const auto& v) { m.insert_or_assign(k, v); });
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
