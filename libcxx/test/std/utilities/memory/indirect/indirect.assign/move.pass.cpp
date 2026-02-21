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

// constexpr indirect& operator=(indirect&& other)
//   noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
//            allocator_traits<Allocator>::is_always_equal::value);

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_allocator.h"
#include "min_allocator.h"

constexpr void test_assignment() {
  { // Move-assigning from an indirect leaves it valueless.
    std::indirect<int> i1(1);
    std::indirect<int> i2(2);

    i1 = std::move(i2); // Both RHS and LHS hold values.
    assert(*i1 == 2);
    assert(i2.valueless_after_move());
    i1 = std::move(i2); // LHS holds value, RHS is valueless.
    assert(i1.valueless_after_move());
    assert(i2.valueless_after_move());
    i1 = std::move(i2); // Both RHS and LHS are valueless.
    assert(i1.valueless_after_move());
    assert(i2.valueless_after_move());
  }
  { // Move assigning to an indirect simply transfers ownership of the held object.
    std::indirect<int, test_allocator<int>> i1(1);
    std::indirect<int, test_allocator<int>> i2(2);
    auto* addr2 = &*i2;
    i1          = std::move(i2);
    assert(i2.valueless_after_move());
    assert(&*i1 == addr2);
    assert(*i1 == 2);
  }
  { // Assigning to an indirect with a different, non-POCMA allocator allocates a new owned object.
    std::indirect<int, test_allocator<int>> i1(std::allocator_arg, test_allocator<int>(1), 1);
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(2), 2);
    auto* addr1 = &*i1;
    auto* addr2 = &*i2;
    i1          = std::move(i2);
    static_assert(!noexcept(i1 = std::move(i2)));
    assert(i2.valueless_after_move());
    assert(&*i1 != addr1);
    assert(&*i1 != addr2);
  }
  { // Assignment returns *this.
    std::indirect<int> i1;
    std::indirect<int> i2;
    std::same_as<std::indirect<int>&> decltype(auto) addr = (i1 = std::move(i2));
    assert(&addr == &i1);
  }
}

void test_assignment_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct MoveThrows {
    int i = 0;
    MoveThrows(int n) : i(n) {}
    MoveThrows(MoveThrows&&) { throw 42; }
    MoveThrows& operator=(MoveThrows&&) { throw 42; }
  };

  std::indirect<MoveThrows, test_allocator<MoveThrows>> i1(std::allocator_arg, test_allocator<MoveThrows>(1), 1);
  std::indirect<MoveThrows, test_allocator<MoveThrows>> i2(std::allocator_arg, test_allocator<MoveThrows>(2), 2);
  auto* addr1 = &*i1;
  auto* addr2 = &*i2;
  try {
    i1 = std::move(i2);
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

struct Immovable {
  Immovable()                            = default;
  Immovable(const Immovable&)            = delete;
  Immovable(Immovable&&)                 = delete;
  Immovable& operator=(const Immovable&) = delete;
  Immovable& operator=(Immovable&&)      = delete;
};

// https://cplusplus.github.io/LWG/issue4251
constexpr void test_lwg4251() {
  { // Move assigning indirect<T> doesn't require T to be copy constructible.
    struct NotCopyConstructible {
      constexpr NotCopyConstructible() = default;
      constexpr NotCopyConstructible(NotCopyConstructible&&) {}
    };
    static_assert(!std::is_copy_constructible_v<NotCopyConstructible>);
    std::indirect<NotCopyConstructible, test_allocator<NotCopyConstructible>> i1;
    std::indirect<NotCopyConstructible, test_allocator<NotCopyConstructible>> i2;
    i1 = std::move(i2);
  }
  { // T doesn't have to be move constructible, as long as the allocator propagates on move.
    using A = other_allocator<Immovable>;
    static_assert(std::allocator_traits<A>::propagate_on_container_move_assignment::value);
    static_assert(!std::allocator_traits<A>::is_always_equal::value);
    std::indirect<Immovable, A> i;
    i = std::move(i);
  }
  { // T doesn't have to be move constructible, as long as the allocator is always equal.
    using A = min_allocator<Immovable>;
    static_assert(!std::allocator_traits<A>::propagate_on_container_move_assignment::value);
    static_assert(std::allocator_traits<A>::is_always_equal::value);
    std::indirect<Immovable, A> i;
    i = std::move(i);
  }
  { // Move assignment with a POCMA allocator simply transfers ownership instead of allocating a new object.
    std::indirect<int, other_allocator<int>> i1(1);
    std::indirect<int, other_allocator<int>> i2(2);
    auto* addr2 = &*i2;
    i1          = std::move(i2);
    assert(i2.valueless_after_move());
    assert(*i1 == 2);
    assert(&*i1 == addr2);
  }
}

constexpr bool test() {
  test_assignment();
  test_lwg4251();

  return true;
}

int main(int, char**) {
  test_assignment_throws();
  test();
  static_assert(test());
  return 0;
}
