//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: sanitizer-new-delete

// template<class T>
//   shared_ptr<T> make_shared_for_overwrite(); // T is not U[]
//
// template<class T>
//   shared_ptr<T> make_shared_for_overwrite(size_t N); // T is U[]

#include <cassert>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "test_macros.h"

template <class T, class... Args>
concept HasMakeSharedForOverwrite =
    requires(Args&&... args) { std::make_shared_for_overwrite<T>(std::forward<Args>(args)...); };

struct Foo {
  int i;
};

// non array
static_assert(HasMakeSharedForOverwrite<int>);
static_assert(HasMakeSharedForOverwrite<Foo>);
static_assert(!HasMakeSharedForOverwrite<int, int>);
static_assert(!HasMakeSharedForOverwrite<Foo, Foo>);

// bounded array
static_assert(HasMakeSharedForOverwrite<int[2]>);
static_assert(!HasMakeSharedForOverwrite<int[2], std::size_t>);
static_assert(!HasMakeSharedForOverwrite<int[2], int>);
static_assert(!HasMakeSharedForOverwrite<int[2], int, int>);
static_assert(HasMakeSharedForOverwrite<Foo[2]>);
static_assert(!HasMakeSharedForOverwrite<Foo[2], std::size_t>);
static_assert(!HasMakeSharedForOverwrite<Foo[2], int>);
static_assert(!HasMakeSharedForOverwrite<Foo[2], int, int>);

// unbounded array
static_assert(HasMakeSharedForOverwrite<int[], std::size_t>);
static_assert(HasMakeSharedForOverwrite<Foo[], std::size_t>);
static_assert(!HasMakeSharedForOverwrite<int[]>);
static_assert(!HasMakeSharedForOverwrite<Foo[]>);
static_assert(!HasMakeSharedForOverwrite<int[], std::size_t, int>);
static_assert(!HasMakeSharedForOverwrite<Foo[], std::size_t, int>);

constexpr char pattern = 0xDE;

void* operator new(std::size_t count) {
  void* ptr = std::malloc(count);
  for (std::size_t i = 0; i < count; ++i) {
    *(reinterpret_cast<char*>(ptr) + i) = pattern;
  }
  return ptr;
}

void* operator new[](std::size_t count) { return ::operator new(count); }

void operator delete(void* ptr) noexcept { std::free(ptr); }

void operator delete[](void* ptr) noexcept { ::operator delete(ptr); }

struct WithDefaultConstructor {
  int i;
  constexpr WithDefaultConstructor() : i(5) {}
};

bool test() {
  // single int
  {
    std::same_as<std::shared_ptr<int>> decltype(auto) ptr = std::make_shared_for_overwrite<int>();
    assert(*(reinterpret_cast<char*>(ptr.get())) == pattern);
  }

  // bounded array int[N]
  {
    std::same_as<std::shared_ptr<int[2]>> decltype(auto) ptr = std::make_shared_for_overwrite<int[2]>();
    assert(*(reinterpret_cast<char*>(&ptr[0])) == pattern);
    assert(*(reinterpret_cast<char*>(&ptr[1])) == pattern);
  }

  // unbounded array int[]
  {
    std::same_as<std::shared_ptr<int[]>> decltype(auto) ptr = std::make_shared_for_overwrite<int[]>(3);
    assert(*(reinterpret_cast<char*>(&ptr[0])) == pattern);
    assert(*(reinterpret_cast<char*>(&ptr[1])) == pattern);
    assert(*(reinterpret_cast<char*>(&ptr[2])) == pattern);
  }

  // single with default constructor
  {
    std::same_as<std::shared_ptr<WithDefaultConstructor>> decltype(auto) ptr =
        std::make_shared_for_overwrite<WithDefaultConstructor>();
    assert(ptr->i == 5);
  }

  // bounded array with default constructor
  {
    std::same_as<std::shared_ptr<WithDefaultConstructor[2]>> decltype(auto) ptr =
        std::make_shared_for_overwrite<WithDefaultConstructor[2]>();
    assert(ptr[0].i == 5);
    assert(ptr[1].i == 5);
  }

  // unbounded array with default constructor
  {
    std::same_as<std::shared_ptr<WithDefaultConstructor[]>> decltype(auto) ptrs =
        std::make_shared_for_overwrite<WithDefaultConstructor[]>(3);
    assert(ptrs[0].i == 5);
    assert(ptrs[1].i == 5);
    assert(ptrs[2].i == 5);
  }

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
