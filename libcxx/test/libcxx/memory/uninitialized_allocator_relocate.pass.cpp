//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the std::__uninitialized_allocator_relocate internal algorithm.

#include <__memory/uninitialized_relocate.h>
#include <cassert>
#include <cstddef>
#include <memory>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class Allocator>
struct Fixture {
  using Traits    = std::allocator_traits<Allocator>;
  using ValueType = typename Traits::value_type;
  using Pointer   = typename Traits::pointer;

  constexpr Fixture(std::size_t n) : size(n) {
    source = allocator.allocate(n);
    for (std::size_t i = 0; i != n; ++i) {
      Traits::construct(allocator, std::to_address(source + i), ValueType(i));
    }

    dest = allocator.allocate(n);
  }

  constexpr void relocated(std::size_t n) { relocated_ = n; }

  constexpr ~Fixture() {
    for (std::size_t i = 0; i != relocated_; ++i) {
      Traits::destroy(allocator, std::to_address(dest + i));
    }
    allocator.deallocate(dest, size);

    for (std::size_t i = relocated_; i != size; ++i) {
      Traits::destroy(allocator, std::to_address(source + i));
    }
    allocator.deallocate(source, size);
  }

  Allocator allocator;
  std::size_t size;
  std::size_t relocated_ = 0;
  Pointer source;
  Pointer dest;
};

struct DestroyTracker {
  explicit DestroyTracker(bool* destroyed) : destroyed_(destroyed) {}
  DestroyTracker(DestroyTracker&& other) : destroyed_(other.destroyed_) { other.destroyed_ = nullptr; }
  ~DestroyTracker() {
    if (destroyed != nullptr)
      *destroyed_ = true;
  }
  bool* destroyed_;
};

template <class Alloc, class Iterator, class OutputIterator>
constexpr void test() {
  using T       = std::allocator_traits<Alloc>::value_type;
  using Pointer = std::allocator_traits<Alloc>::pointer;

  // Relocate an empty range
  {
    Fixture<Alloc> t(10);

    OutputIterator res = std::__uninitialized_allocator_relocate(
        t.allocator,
        Iterator(std::to_address(t.source)),
        Iterator(std::to_address(t.source)),
        OutputIterator(std::to_address(t.dest)));
    assert(res == OutputIterator(std::to_address(t.dest)));
    t.relocated(0);

    for (int i = 0; i != 10; ++i) {
      assert(t.source[i] == T(i));
    }
  }

  // Range of size 1
  {
    Fixture<Alloc> t(10);

    OutputIterator res = std::__uninitialized_allocator_relocate(
        t.allocator,
        Iterator(std::to_address(t.source)),
        Iterator(std::to_address(t.source + 1)),
        OutputIterator(std::to_address(t.dest)));
    assert(res == OutputIterator(std::to_address(t.dest + 1)));
    t.relocated(1);

    assert(t.dest[0] == T(0));
    assert(t.source[1] == T(1));
    assert(t.source[2] == T(2));
    // ...
  }

  // Range of normal size
  {
    Fixture<Alloc> t(10);

    OutputIterator res = std::__uninitialized_allocator_relocate(
        t.allocator,
        Iterator(std::to_address(t.source)),
        Iterator(std::to_address(t.source + 10)),
        OutputIterator(std::to_address(t.dest)));
    assert(res == OutputIterator(std::to_address(t.dest + 10)));
    t.relocated(10);

    for (int i = 0; i != 10; ++i) {
      assert(t.dest[i] == T(i));
    }
  }

  // Relocate with some overlap between the input and the output range
  {
    Alloc allocator;
    Pointer buff = allocator.allocate(10);
    // x x x x x 5 6 7 8 9
    for (std::size_t i = 5; i != 10; ++i) {
      std::allocator_traits<Alloc>::construct(allocator, std::to_address(buff + i), T(i));
    }

    OutputIterator res = std::__uninitialized_allocator_relocate(
        allocator,
        Iterator(std::to_address(buff + 5)),
        Iterator(std::to_address(buff + 10)),
        OutputIterator(std::to_address(buff)));
    assert(res == OutputIterator(std::to_address(buff + 5)));

    for (int i = 0; i != 5; ++i) {
      assert(buff[i] == T(i + 5));
      std::allocator_traits<Alloc>::destroy(allocator, std::to_address(buff + i));
    }
    allocator.deallocate(buff, 10);
  }

  // Test throwing an exception while we're relocating
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    // With some overlap between the input and the output
    {
      bool destroyed[10] = {false, false, ...};
      DestroyTracker elements[] = {&destroyed[0], &destroyed[1], ...};
      try {
        std::__uninitialized_allocator_relocate(
            allocator,
            Iterator(std::to_address(buff + 5)),
            Iterator(std::to_address(buff + 10)),
            OutputIterator(std::to_address(buff)));
      } catch (...) {
        // TODO: ensure we destroyed everything
      }
    }

    // Without overlap
    {
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

struct NotTriviallyRelocatable {
  constexpr explicit NotTriviallyRelocatable(int i) : value_(i) {}
  constexpr NotTriviallyRelocatable(NotTriviallyRelocatable&& other) : value_(other.value_) { other.value_ = -1; }
  constexpr friend bool operator==(NotTriviallyRelocatable const& a, NotTriviallyRelocatable const& b) {
    return a.value_ == b.value_;
  }

  int value_;
};

template <class T>
struct ConsructAllocator : std::allocator<T> {
  template < class... Args>
  constexpr void construct(T* loc, Args&&... args) {
    std::construct_at(loc, std::forward<Args>(args)...);
  }
};

template <class T>
struct DestroyAllocator : std::allocator<T> {
  constexpr void destroy(T* loc) { std::destroy_at(loc); }
};

constexpr bool tests() {
  test<std::allocator<int>, cpp17_input_iterator<int*>, forward_iterator<int*>>();
  test<std::allocator<int>, contiguous_iterator<int*>, contiguous_iterator<int*>>();
  test<min_allocator<int>, cpp17_input_iterator<int*>, forward_iterator<int*>>();
  test<std::allocator<NotTriviallyRelocatable>,
       cpp17_input_iterator<NotTriviallyRelocatable*>,
       forward_iterator<NotTriviallyRelocatable*>>();
  test<ConsructAllocator<NotTriviallyRelocatable>,
       cpp17_input_iterator<NotTriviallyRelocatable*>,
       forward_iterator<NotTriviallyRelocatable*>>();
  test<DestroyAllocator<NotTriviallyRelocatable>,
       cpp17_input_iterator<NotTriviallyRelocatable*>,
       forward_iterator<NotTriviallyRelocatable*>>();
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests(), "");
#endif
  return 0;
}
