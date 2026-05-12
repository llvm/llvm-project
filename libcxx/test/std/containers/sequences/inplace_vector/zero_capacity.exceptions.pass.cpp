//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

// Clang's -Wunreachable-code does not model exceptional exits from [[noreturn]] members
// (e.g. at() in this specialization), flagging the catch handlers below.
// ADDITIONAL_COMPILE_FLAGS: -Wno-unreachable-code

// <inplace_vector>

#include <cassert>
#include <inplace_vector>
#include <new>
#include <stdexcept>
#include <utility>

#include "common.h"
#include "test_macros.h"

using C = std::inplace_vector<int, 0>;

void test_throwing_members() {
  int a[] = {1, 2};

  assert_throws_bad_alloc([] { C c(1); });
  assert_throws_bad_alloc([] { C c(1, 42); });
  assert_throws_bad_alloc([&] { C c(a, a + 2); });
  assert_throws_bad_alloc([&] { C c(std::from_range, a); });
  assert_throws_bad_alloc([] { C c{1, 2}; });

  C c;
  assert_throws_bad_alloc([&] { c = {1, 2}; });
  assert_throws_bad_alloc([&] { c.assign(a, a + 2); });
  assert_throws_bad_alloc([&] { c.assign_range(a); });
  assert_throws_bad_alloc([&] { c.assign(2, 42); });
  assert_throws_bad_alloc([&] { c.assign({1, 2}); });

  assert_throws_bad_alloc([&] { c.resize(1); });
  assert_throws_bad_alloc([&] { c.resize(1, 42); });
  assert_throws_bad_alloc([&] { c.reserve(1); });
  assert_throws_bad_alloc([&] { C::reserve(1); });

  int value = 42;
  assert_throws_bad_alloc([&] { c.emplace_back(42); });
  assert_throws_bad_alloc([&] { c.push_back(value); });
  assert_throws_bad_alloc([&] { c.push_back(42); });
  assert_throws_bad_alloc([&] { c.append_range(a); });

  assert_throws_bad_alloc([&] { c.emplace(c.begin(), 42); });
  assert_throws_bad_alloc([&] { c.insert(c.begin(), value); });
  assert_throws_bad_alloc([&] { c.insert(c.begin(), 42); });
  assert_throws_bad_alloc([&] { c.insert(c.begin(), 2, 42); });
  assert_throws_bad_alloc([&] { c.insert(c.begin(), a, a + 2); });
  assert_throws_bad_alloc([&] { c.insert_range(c.begin(), a); });
  assert_throws_bad_alloc([&] { c.insert(c.begin(), {1, 2}); });

  assert(c.empty());

  bool threw = false;
  try {
    (void)c.at(0);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)std::as_const(c).at(0);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);
}

constexpr bool test_non_throwing_members() {
  int a[] = {1, 2};

  // Requests for zero elements and empty inputs must not throw.
  C c(0);
  C c2(0, 42);
  C c3(a, a);
  C c4{};
  assert(c2.empty() && c3.empty() && c4.empty());
  c = {};
  c.assign(a, a);
  c.assign(0, 42);
  c.assign(std::initializer_list<int>{});
  c.resize(0);
  c.resize(0, 42);
  c.reserve(0);
  c.shrink_to_fit();

  assert(c.insert(c.begin(), 0, 42) == c.begin());
  assert(c.insert(c.begin(), a, a) == c.begin());
  assert(c.insert(c.begin(), std::initializer_list<int>{}) == c.begin());
  assert(c.erase(c.begin(), c.begin()) == c.begin());

  // The try_* forms report a full container instead of throwing.
  int value = 42;
  assert(!c.try_emplace_back(42).has_value());
  assert(!c.try_push_back(value).has_value());
  assert(!c.try_push_back(42).has_value());

  assert(c.empty());

  return true;
}

int main(int, char**) {
  test_throwing_members();

  test_non_throwing_members();
  static_assert(test_non_throwing_members());

  return 0;
}
