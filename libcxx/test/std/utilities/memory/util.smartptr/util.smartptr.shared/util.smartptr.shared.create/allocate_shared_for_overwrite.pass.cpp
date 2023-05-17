//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class A>
//   shared_ptr<T> make_shared_for_overwrite(const A& a); // T is not U[]
//
// template<class T, class A>
//   shared_ptr<T> make_shared_for_overwrite(const A& a, size_t N); // T is U[]

#include <cassert>
#include <concepts>
#include <cstring>
#include <memory>
#include <utility>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T, class... Args>
concept HasAllocateSharedForOverwrite =
    requires(Args&&... args) { std::allocate_shared_for_overwrite<T>(std::forward<Args>(args)...); };

struct Foo {
  int i;
};

// non array
static_assert(!HasAllocateSharedForOverwrite<int>);
static_assert(!HasAllocateSharedForOverwrite<Foo>);
static_assert(HasAllocateSharedForOverwrite<int, bare_allocator<void>>);
static_assert(HasAllocateSharedForOverwrite<Foo, bare_allocator<void>>);
static_assert(!HasAllocateSharedForOverwrite<int, bare_allocator<void>, std::size_t>);
static_assert(!HasAllocateSharedForOverwrite<Foo, bare_allocator<void>, std::size_t>);

// bounded array
static_assert(!HasAllocateSharedForOverwrite<int[2]>);
static_assert(!HasAllocateSharedForOverwrite<Foo[2]>);
static_assert(HasAllocateSharedForOverwrite<int[2], bare_allocator<void>>);
static_assert(HasAllocateSharedForOverwrite<Foo[2], bare_allocator<void>>);
static_assert(!HasAllocateSharedForOverwrite<int[2], bare_allocator<void>, std::size_t>);
static_assert(!HasAllocateSharedForOverwrite<Foo[2], bare_allocator<void>, std::size_t>);

// unbounded array
static_assert(!HasAllocateSharedForOverwrite<int[]>);
static_assert(!HasAllocateSharedForOverwrite<Foo[]>);
static_assert(!HasAllocateSharedForOverwrite<int[], bare_allocator<void>>);
static_assert(!HasAllocateSharedForOverwrite<Foo[], bare_allocator<void>>);
static_assert(HasAllocateSharedForOverwrite<int[], bare_allocator<void>, std::size_t>);
static_assert(HasAllocateSharedForOverwrite<Foo[], bare_allocator<void>, std::size_t>);

struct WithDefaultCtor {
  int i;
  WithDefaultCtor() : i(42) {}
};

template <class Alloc>
void testDefaultConstructor() {
  // single
  {
    std::same_as<std::shared_ptr<WithDefaultCtor>> auto ptr =
        std::allocate_shared_for_overwrite<WithDefaultCtor>(Alloc{});
    assert(ptr->i == 42);
  }

  // bounded array
  {
    std::same_as<std::shared_ptr<WithDefaultCtor[2]>> auto ptr =
        std::allocate_shared_for_overwrite<WithDefaultCtor[2]>(Alloc{});
    assert(ptr[0].i == 42);
    assert(ptr[1].i == 42);
  }

  // unbounded array
  {
    std::same_as<std::shared_ptr<WithDefaultCtor[]>> auto ptr =
        std::allocate_shared_for_overwrite<WithDefaultCtor[]>(Alloc{}, 3);
    assert(ptr[0].i == 42);
    assert(ptr[1].i == 42);
    assert(ptr[2].i == 42);
  }
}

void testTypeWithDefaultCtor() {
  testDefaultConstructor<test_allocator<WithDefaultCtor>>();
  testDefaultConstructor<min_allocator<WithDefaultCtor>>();
  testDefaultConstructor<bare_allocator<WithDefaultCtor>>();
}

struct CountDestructions {
  int* destructions_;
  constexpr CountDestructions() = default;
  constexpr CountDestructions(int* d) : destructions_(d) { }
  constexpr ~CountDestructions() { ++*destructions_; }
};

void testAllocatorOperationsCalled() {
  // single
  {
    test_allocator_statistics alloc_stats;
    int destructions = 0;
    {
      [[maybe_unused]] std::same_as<std::shared_ptr<CountDestructions>> auto ptr =
          std::allocate_shared_for_overwrite<CountDestructions>(test_allocator<void>{&alloc_stats});
      std::construct_at<CountDestructions>(ptr.get(), &destructions);
      assert(alloc_stats.alloc_count == 1);
      assert(alloc_stats.construct_count == 0);
    }
    assert(destructions == 1);
    assert(alloc_stats.destroy_count == 0);
    assert(alloc_stats.alloc_count == 0);
  }

  // bounded array
  {
    test_allocator_statistics alloc_stats;
    int destructions = 0;
    {
      [[maybe_unused]] std::same_as<std::shared_ptr<CountDestructions[2]>> auto ptr =
          std::allocate_shared_for_overwrite<CountDestructions[2]>(test_allocator<void>{&alloc_stats});
      std::construct_at<CountDestructions>(ptr.get() + 0, &destructions);
      std::construct_at<CountDestructions>(ptr.get() + 1, &destructions);
      assert(alloc_stats.alloc_count == 1);
      assert(alloc_stats.construct_count == 0);
    }
    assert(destructions == 2);
    assert(alloc_stats.destroy_count == 0);
    assert(alloc_stats.alloc_count == 0);
  }

  // unbounded array
  {
    test_allocator_statistics alloc_stats;
    int destructions = 0;
    {
      [[maybe_unused]] std::same_as<std::shared_ptr<CountDestructions[]>> auto ptr =
          std::allocate_shared_for_overwrite<CountDestructions[]>(test_allocator<void>{&alloc_stats}, 3);
      std::construct_at<CountDestructions>(ptr.get() + 0, &destructions);
      std::construct_at<CountDestructions>(ptr.get() + 1, &destructions);
      std::construct_at<CountDestructions>(ptr.get() + 2, &destructions);
      assert(alloc_stats.alloc_count == 1);
      assert(alloc_stats.construct_count == 0);
    }
    assert(destructions == 3);
    assert(alloc_stats.destroy_count == 0);
    assert(alloc_stats.alloc_count == 0);
  }
}

template <class T>
struct AllocatorWithPattern {
  constexpr static char pattern = 0xDE;

  using value_type = T;

  AllocatorWithPattern() = default;

  template <class U>
  AllocatorWithPattern(AllocatorWithPattern<U>) noexcept {}

  T* allocate(std::size_t n) {
    void* ptr = ::operator new(n * sizeof(T));
    for (std::size_t i = 0; i < n * sizeof(T); ++i) {
      *(reinterpret_cast<char*>(ptr) + i) = pattern;
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) { return ::operator delete(static_cast<void*>(p)); }
};

void testNotValueInitialized() {
  // single int
  {
    std::same_as<std::shared_ptr<int>> decltype(auto) ptr =
        std::allocate_shared_for_overwrite<int>(AllocatorWithPattern<int>{});
    assert(*(reinterpret_cast<char*>(ptr.get())) == AllocatorWithPattern<int>::pattern);
  }

  // bounded array int[N]
  {
    std::same_as<std::shared_ptr<int[2]>> decltype(auto) ptr =
        std::allocate_shared_for_overwrite<int[2]>(AllocatorWithPattern<int>{});
    assert(*(reinterpret_cast<char*>(&ptr[0])) == AllocatorWithPattern<int>::pattern);
    assert(*(reinterpret_cast<char*>(&ptr[1])) == AllocatorWithPattern<int>::pattern);
  }

  // unbounded array int[]
  {
    std::same_as<std::shared_ptr<int[]>> decltype(auto) ptr =
        std::allocate_shared_for_overwrite<int[]>(AllocatorWithPattern<int>{}, 3);
    assert(*(reinterpret_cast<char*>(&ptr[0])) == AllocatorWithPattern<int>::pattern);
    assert(*(reinterpret_cast<char*>(&ptr[1])) == AllocatorWithPattern<int>::pattern);
    assert(*(reinterpret_cast<char*>(&ptr[2])) == AllocatorWithPattern<int>::pattern);
  }
}

void test() {
  testTypeWithDefaultCtor();
  testAllocatorOperationsCalled();
  testNotValueInitialized();
}

int main(int, char**) {
  test();

  return 0;
}
