//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <class T>
// T* start_lifetime_as(void* p) noexcept;
//
// template <class T>
// const T* start_lifetime_as(const void* p) noexcept;
//
// template <class T>
// volatile T* start_lifetime_as(volatile void* p) noexcept;
//
// template <class T>
// const volatile T* start_lifetime_as(const volatile void* p) noexcept;
//
// template <class T>
// T* start_lifetime_as_array(void* p, size_t n) noexcept;
//
// template <class T>
// const T* start_lifetime_as_array(const void* p, size_t n) noexcept;
//
// template <class T>
// volatile T* start_lifetime_as_array(volatile void* p, size_t n) noexcept;
//
// template <class T>
// const volatile T* start_lifetime_as_array(const volatile void* p, size_t n) noexcept;

#include <memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct S {
  int x;
  double y;
};

template <class T>
void test_start_lifetime_as() {
  alignas(T) char buffer[sizeof(T)];

  {
    T* ptr = std::start_lifetime_as<T>(buffer);
    ASSERT_SAME_TYPE(decltype(ptr), T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as<T>(buffer));
    assert(ptr == reinterpret_cast<T*>(buffer));
  }

  {
    const T* cptr = std::start_lifetime_as<T>(static_cast<const void*>(buffer));
    ASSERT_SAME_TYPE(decltype(cptr), const T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as<T>(static_cast<const void*>(buffer)));
    assert(cptr == reinterpret_cast<const T*>(buffer));
  }

  {
    volatile T* vptr = std::start_lifetime_as<T>(static_cast<volatile void*>(buffer));
    ASSERT_SAME_TYPE(decltype(vptr), volatile T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as<T>(static_cast<volatile void*>(buffer)));
    assert(vptr == reinterpret_cast<volatile T*>(buffer));
  }

  {
    const volatile T* cvptr = std::start_lifetime_as<T>(static_cast<const volatile void*>(buffer));
    ASSERT_SAME_TYPE(decltype(cvptr), const volatile T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as<T>(static_cast<const volatile void*>(buffer)));
    assert(cvptr == reinterpret_cast<const volatile T*>(buffer));
  }
}

template <class T>
void test_start_lifetime_as_array() {
  constexpr size_t count = 5;
  alignas(T) char buffer[sizeof(T) * count];

  {
    T* ptr = std::start_lifetime_as_array<T>(buffer, count);
    ASSERT_SAME_TYPE(decltype(ptr), T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as_array<T>(buffer, count));
    assert(ptr == reinterpret_cast<T*>(buffer));
  }

  {
    const T* cptr = std::start_lifetime_as_array<T>(static_cast<const void*>(buffer), count);
    ASSERT_SAME_TYPE(decltype(cptr), const T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as_array<T>(static_cast<const void*>(buffer), count));
    assert(cptr == reinterpret_cast<const T*>(buffer));
  }

  {
    volatile T* vptr = std::start_lifetime_as_array<T>(static_cast<volatile void*>(buffer), count);
    ASSERT_SAME_TYPE(decltype(vptr), volatile T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as_array<T>(static_cast<volatile void*>(buffer), count));
    assert(vptr == reinterpret_cast<volatile T*>(buffer));
  }

  {
    const volatile T* cvptr = std::start_lifetime_as_array<T>(static_cast<const volatile void*>(buffer), count);
    ASSERT_SAME_TYPE(decltype(cvptr), const volatile T*);
    ASSERT_NOEXCEPT(std::start_lifetime_as_array<T>(static_cast<const volatile void*>(buffer), count));
    assert(cvptr == reinterpret_cast<const volatile T*>(buffer));
  }
}

int main(int, char**) {
  test_start_lifetime_as<char>();
  test_start_lifetime_as<int>();
  test_start_lifetime_as<double>();
  test_start_lifetime_as<S>();

  test_start_lifetime_as_array<char>();
  test_start_lifetime_as_array<int>();
  test_start_lifetime_as_array<double>();
  test_start_lifetime_as_array<S>();

  return 0;
}