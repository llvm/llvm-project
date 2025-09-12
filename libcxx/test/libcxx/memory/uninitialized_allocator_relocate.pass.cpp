//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the std::__uninitialized_allocator_relocate internal algorithm.

// This test is impossible to write without std::to_address and std::construct_at
// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__memory/allocator_relocation.h>
#include <cassert>
#include <cstddef>
#include <memory>

#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

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

struct AliveTracker {
  explicit AliveTracker(int v, bool do_throw, bool* alive) : value(v), do_throw_(do_throw), alive_(alive) {
    if (alive_ != nullptr)
      *alive_ = true;
  }
  AliveTracker(AliveTracker&& other) : value(other.value), do_throw_(other.do_throw_), alive_(other.alive_) {
    other.alive_ = nullptr;
    if (do_throw_) {
      if (alive_ != nullptr)
        *alive_ = false; // we failed to be constructed
      throw 42;
    }
  }
  ~AliveTracker() {
    if (alive_ != nullptr) {
      assert(*alive_ == true && "double destroy?");
      *alive_ = false;
    }
  }
  int value;
  bool do_throw_;
  bool* alive_;
};

template <template <class...> class Alloc,
          template <class...> class Iter,
          template <class...> class OutputIter,
          class T>
constexpr void test() {
  using Allocator      = Alloc<T>;
  using Iterator       = Iter<T*>;
  using OutputIterator = OutputIter<T*>;

  // Relocate an empty range
  {
    Fixture<Allocator> t(10);

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
    Fixture<Allocator> t(10);

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
    Fixture<Allocator> t(10);

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
    Allocator allocator;
    auto buff = allocator.allocate(10);
    // x x x x x 5 6 7 8 9
    for (std::size_t i = 5; i != 10; ++i) {
      std::allocator_traits<Allocator>::construct(allocator, std::to_address(buff + i), T(i));
    }

    OutputIterator res = std::__uninitialized_allocator_relocate(
        allocator,
        Iterator(std::to_address(buff + 5)),
        Iterator(std::to_address(buff + 10)),
        OutputIterator(std::to_address(buff)));
    assert(res == OutputIterator(std::to_address(buff + 5)));

    for (int i = 0; i != 5; ++i) {
      assert(buff[i] == T(i + 5));
      std::allocator_traits<Allocator>::destroy(allocator, std::to_address(buff + i));
    }
    allocator.deallocate(buff, 10);
  }

  // Test throwing an exception while we're relocating
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    {
      using A     = Alloc<AliveTracker>;
      using It    = Iter<AliveTracker*>;
      using OutIt = OutputIter<AliveTracker*>;
      // With some overlap between the input and the output
      {
        // Note: this is just storage for the bools and doesn't actually represent which elements are
        //       alive in the right order when they are moved around.
        bool alive[10]         = {false, false, false, false, false, false, false, false, false, false};
        bool throw_on_move[10] = {false, false, false, false, false, false, true, false, false, false};
        A allocator;
        auto buff = allocator.allocate(10);
        // x x x 3 4 5 6 7 8 9
        for (std::size_t i = 3; i != 10; ++i) {
          std::allocator_traits<A>::construct(allocator, std::to_address(buff + i), i, throw_on_move[i], &alive[i]);
        }

        // test the test
        for (std::size_t i = 3; i != 10; ++i)
          assert(alive[i]);

        try {
          std::__uninitialized_allocator_relocate(
              allocator, It(std::to_address(buff + 3)), It(std::to_address(buff + 10)), OutIt(std::to_address(buff)));
          assert(false && "there should have been an exception");
        } catch (...) {
        }

        for (std::size_t i = 0; i != 10; ++i)
          assert(!alive[i]);

        allocator.deallocate(buff, 10);
      }

      // Without overlap
      {
        bool alive[10]         = {false, false, false, false, false, false, false, false, false, false};
        bool throw_on_move[10] = {false, false, false, false, false, false, false, true, false, false};
        A allocator;
        auto buff = allocator.allocate(10);
        // x x x x x 5 6 7 8 9
        for (std::size_t i = 5; i != 10; ++i) {
          std::allocator_traits<A>::construct(allocator, std::to_address(buff + i), i, throw_on_move[i], &alive[i]);
        }

        // test the test
        for (std::size_t i = 5; i != 10; ++i)
          assert(alive[i]);

        try {
          std::__uninitialized_allocator_relocate(
              allocator, It(std::to_address(buff + 5)), It(std::to_address(buff + 10)), OutIt(std::to_address(buff)));
          assert(false && "there should have been an exception");
        } catch (...) {
        }

        for (std::size_t i = 0; i != 10; ++i)
          assert(!alive[i]);

        allocator.deallocate(buff, 10);
      }
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
static_assert(!std::__libcpp_is_trivially_relocatable<NotTriviallyRelocatable>::value, "");

template <class T>
struct ConstructAllocator : std::allocator<T> {
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
  types::for_each(types::type_list<int, NotTriviallyRelocatable>{}, []<class T> {
    test<std::allocator, forward_iterator, forward_iterator, T>();
    test<std::allocator, contiguous_iterator, contiguous_iterator, T>();

    test<min_allocator, forward_iterator, forward_iterator, T>();
    test<min_allocator, contiguous_iterator, contiguous_iterator, T>();

    test<ConstructAllocator, forward_iterator, forward_iterator, T>();
    test<ConstructAllocator, contiguous_iterator, contiguous_iterator, T>();

    test<DestroyAllocator, forward_iterator, forward_iterator, T>();
    test<DestroyAllocator, contiguous_iterator, contiguous_iterator, T>();
  });
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests(), "");
#endif
  return 0;
}
