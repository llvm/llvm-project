//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// constexpr void swap(indirect& other)
//   noexcept(allocator_traits<Allocator>::propagate_on_container_swap::value ||
//            allocator_traits<Allocator>::is_always_equal::value);

// constexpr void swap(indirect& lhs, indirect& rhs) noexcept(noexcept(lhs.swap(rhs)));

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_allocator.h"
#include "test_convertible.h"

template <class T, bool POCS>
struct pocs_allocator {
  using value_type                  = T;
  using propagate_on_container_swap = std::bool_constant<POCS>;

  template <typename U>
  struct rebind {
    using other = pocs_allocator<U, POCS>;
  };

  int data = 0;

  constexpr pocs_allocator(int i) : data(i) {}
  constexpr T* allocate(size_t n) { return std::allocator<T>().allocate(n); }
  constexpr void deallocate(T* ptr, size_t n) { return std::allocator<T>().deallocate(ptr, n); }

  friend constexpr bool operator==(pocs_allocator, pocs_allocator) { return true; };

  friend constexpr void swap(pocs_allocator& lhs, pocs_allocator& rhs) { std::swap(lhs.data, rhs.data); }
};

constexpr void test_swap_noexcept() {
  std::indirect<int> i;
  static_assert(noexcept(swap(i, i)));
  static_assert(noexcept(i.swap(i)));
}

constexpr void test_swap() {
  {
    std::indirect<int> i1(1);
    std::indirect<int> i2(2);
    swap(i1, i2);
    assert(*i1 == 2);
    assert(*i2 == 1);
  }
  {
    using A = pocs_allocator<int, true>;
    std::indirect<int, A> i1(std::allocator_arg, A(1), 1);
    std::indirect<int, A> i2(std::allocator_arg, A(2), 2);
    swap(i1, i2);
    assert(*i1 == 2);
    assert(*i2 == 1);
    assert(i1.get_allocator().data == 2);
    assert(i2.get_allocator().data == 1);
    static_assert(noexcept(swap(i1, i2)));
  }
  {
    using A = pocs_allocator<int, false>;
    std::indirect<int, A> i1(std::allocator_arg, A(1), 1);
    std::indirect<int, A> i2(std::allocator_arg, A(2), 2);
    swap(i1, i2);
    assert(*i1 == 2);
    assert(*i2 == 1);
    assert(i1.get_allocator().data == 1);
    assert(i2.get_allocator().data == 2);
    static_assert(!noexcept(swap(i1, i2)));
  }
  struct Incomplete;
  { // Swapping incomplete types is valid.
    (void)([](std::indirect<Incomplete>& i) {
      swap(i, i);
      i.swap(i);
    });
  }
}

constexpr bool test() {
  test_swap_noexcept();
  test_swap();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
