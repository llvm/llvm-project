//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// yield_value(yielded);
// yield_value(const remove_reference_t<yielded>&);

#include <generator>

#include <cassert>
#include <initializer_list>
#include <memory>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct Copyable {
  int value;
  int* copies;
  Copyable(int v, int& count) : value(v), copies(&count) {}
  Copyable(const Copyable& other) : value(other.value), copies(other.copies) { ++*copies; }
  Copyable(Copyable&& other) : value(std::exchange(other.value, -1)), copies(other.copies) {}
};

std::generator<Copyable> copy_and_move(Copyable& value) {
  co_yield value;
  assert(value.value == 42);
  co_yield std::move(value);
  assert(value.value == -1);
}

void test_copy_and_move() {
  int copies = 0;
  Copyable value(42, copies);
  auto g = copy_and_move(value);
  auto i = g.begin();
  assert(copies == 1);
  Copyable first = *i;
  assert(first.value == 42 && value.value == 42 && copies == 1);
  ++i;
  Copyable second = *i;
  assert(second.value == 42 && value.value == -1 && copies == 1);
  ++i;
  assert(i == g.end());
}

std::generator<int&> references(int& value) {
  co_yield value;
  assert(value == 7);
}

struct Immovable {
  Immovable()                 = default;
  Immovable(const Immovable&) = delete;
  Immovable(Immovable&&)      = delete;
};

std::generator<const Immovable&> immovable() { co_yield Immovable{}; }
std::generator<MoveOnly> move_only() { co_yield MoveOnly(42); }

// The stored object is direct-non-list-initialized, including when its copy constructor is explicit.
struct ExplicitCopy {
  ExplicitCopy()                             = default;
  explicit ExplicitCopy(const ExplicitCopy&) = default;
};
std::generator<ExplicitCopy&&> explicit_copy() {
  const ExplicitCopy value;
  co_yield value;
}

struct ListConstructor {
  int value                               = 42;
  ListConstructor()                       = default;
  ListConstructor(const ListConstructor&) = default;
  ListConstructor(std::initializer_list<int>) : value(-1) {}
  operator int() const { return value; }
};
std::generator<ListConstructor> list_constructor() {
  const ListConstructor value;
  co_yield value;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct ThrowingCopy {
  ThrowingCopy() = default;
  ThrowingCopy(const ThrowingCopy&) { throw 42; }
};
std::generator<ThrowingCopy> throwing_copy() {
  const ThrowingCopy value;
  co_yield value;
}
#endif

int main(int, char**) {
  test_copy_and_move();
  {
    int value = 1;
    auto g    = references(value);
    auto i    = g.begin();
    assert(std::addressof(*i) == &value);
    *i = 7;
    ++i;
    assert(i == g.end());
  }
  {
    auto g                                  = immovable();
    auto i                                  = g.begin();
    [[maybe_unused]] const Immovable& value = *i;
    ++i;
    assert(i == g.end());
  }
  {
    auto g         = move_only();
    auto i         = g.begin();
    MoveOnly value = *i;
    assert(value.get() == 42);
    ++i;
    assert(i == g.end());
  }
  {
    auto g = explicit_copy();
    auto i = g.begin();
    ++i;
    assert(i == g.end());
  }
  {
    auto g = list_constructor();
    auto i = g.begin();
    assert((*i).value == 42);
    ++i;
    assert(i == g.end());
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    auto g = throwing_copy();
    (void)g.begin();
    assert(false);
  } catch (int value) {
    assert(value == 42);
  }
#endif
  return 0;
}
