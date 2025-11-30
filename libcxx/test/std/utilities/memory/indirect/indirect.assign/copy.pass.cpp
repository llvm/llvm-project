//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// constexpr indirect& operator=(const indirect& other);

#include <cassert>
#include <type_traits>
#include <memory>

#include "test_convertible.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "archetypes.h"

constexpr void test_assignment() {
  { // Assigning from a valueless indirect destroys the owned object, if any.
    std::indirect<int> i1;
    std::indirect<int> i2;

    auto(std::move(i2));
    i1 = i2;
    assert(i1.valueless_after_move());
    assert(i2.valueless_after_move());
    i1 = i2;
    assert(i1.valueless_after_move());
    assert(i2.valueless_after_move());
  }
  { // Assigning to an indirect that already owns an object doesn't allocate a new object.
    test_allocator_statistics stats;
    std::indirect<int, test_allocator<int>> i1(std::allocator_arg, test_allocator<int>(&stats), 1);
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(&stats), 2);
    assert(stats.construct_count == 2);
    auto* addr_before = &*i1;
    i1                = i2;
    assert(addr_before == &*i1);
    assert(stats.construct_count == 2);
    assert(*i1 == 2);
    assert(*i2 == 2);
  }
  { // Assigning to an indirect with a different allocator allocates a new owned object.
    test_allocator_statistics stats;
    std::indirect<int, test_allocator<int>> i1(std::allocator_arg, test_allocator<int>(1, &stats), 1);
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(2, &stats), 2);
    assert(stats.construct_count == 2);
    auto* addr_before = &*i1;
    i1                = i2;
    assert(addr_before != &*i1);
    assert(stats.construct_count == 3);
    assert(*i1 == 2);
    assert(*i2 == 2);
  }
  { // Assigning to a valueless indirect allocates a new owned object.
    test_allocator_statistics stats;
    std::indirect<int, test_allocator<int>> i1(std::allocator_arg, test_allocator<int>(&stats), 1);
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(&stats), 2);
    assert(stats.construct_count == 2);
    auto(std::move(i1));
    i1 = i2;
    assert(*i1 == 2);
    assert(*i2 == 2);
    assert(stats.construct_count == 3);
  }
  { // Assignment returns *this.
    std::indirect<int> i1;
    const std::indirect<int> i2;
    std::same_as<std::indirect<int>&> decltype(auto) addr = (i1 = i2);
    assert(&addr == &i1);
  }
}

void test_assignment_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct CopyingThrows {
    int i = 0;
    CopyingThrows(int n) : i(n) {}
    CopyingThrows(const CopyingThrows&) { throw 42; }
    CopyingThrows& operator=(const CopyingThrows&) { throw 42; }
  };

  std::indirect<CopyingThrows, test_allocator<CopyingThrows>> i1(
      std::allocator_arg, test_allocator<CopyingThrows>(1), 1);
  std::indirect<CopyingThrows, test_allocator<CopyingThrows>> i2(
      std::allocator_arg, test_allocator<CopyingThrows>(2), 2);
  auto* addr1 = &*i1;
  auto* addr2 = &*i2;
  try {
    i1 = i2;
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }
  assert(addr1 == &*i1);
  assert(addr2 == &*i2);
  assert(i1->i == 1);
  assert(i2->i == 2);
  assert(i1.get_allocator().get_data() == 1);
  assert(i2.get_allocator().get_data() == 2);
#endif
}

constexpr bool test() {
  test_assignment();

  return true;
}

int main(int, char**) {
  test_assignment_throws();
  test();
  static_assert(test());
  return 0;
}
