//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N>
// struct array

// Make sure std::array<T, N> has the correct object size and alignment.
// This test is mostly meant to catch subtle ABI-breaking regressions.

// Ignore error about requesting a large alignment not being ABI compatible with older AIX systems.
#if defined(_AIX)
#  pragma clang diagnostic ignored "-Waix-compat"
#endif

#include <array>
#include <cstddef>
#include <type_traits>

#ifdef _LIBCPP_VERSION
#  include <__type_traits/datasizeof.h>
#endif

#include "test_macros.h"

template <class T, std::size_t Size>
struct MyArray {
  T elems[Size];
};

template <class T>
void test_type() {
  {
    using Array = std::array<T, 0>;
    LIBCPP_STATIC_ASSERT(sizeof(Array) == sizeof(T), "");
    LIBCPP_STATIC_ASSERT(TEST_ALIGNOF(Array) == TEST_ALIGNOF(T), "");
    LIBCPP_STATIC_ASSERT(sizeof(Array) == sizeof(T[1]), "");
    LIBCPP_STATIC_ASSERT(sizeof(Array) == sizeof(MyArray<T, 1>), "");
    LIBCPP_STATIC_ASSERT(TEST_ALIGNOF(Array) == TEST_ALIGNOF(MyArray<T, 1>), "");
    static_assert(!std::is_empty<Array>::value, "");

    // Make sure empty arrays don't have padding bytes
    LIBCPP_STATIC_ASSERT(std::__datasizeof_v<Array> == sizeof(Array), "");
  }

  {
    using Array = std::array<T, 1>;
    static_assert(sizeof(Array) == sizeof(T), "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(T), "");
    static_assert(sizeof(Array) == sizeof(T[1]), "");
    static_assert(sizeof(Array) == sizeof(MyArray<T, 1>), "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(MyArray<T, 1>), "");
    static_assert(!std::is_empty<Array>::value, "");
  }

  {
    using Array = std::array<T, 2>;
    static_assert(sizeof(Array) == sizeof(T) * 2, "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(T), "");
    static_assert(sizeof(Array) == sizeof(T[2]), "");
    static_assert(sizeof(Array) == sizeof(MyArray<T, 2>), "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(MyArray<T, 2>), "");
    static_assert(!std::is_empty<Array>::value, "");
  }

  {
    using Array = std::array<T, 3>;
    static_assert(sizeof(Array) == sizeof(T) * 3, "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(T), "");
    static_assert(sizeof(Array) == sizeof(T[3]), "");
    static_assert(sizeof(Array) == sizeof(MyArray<T, 3>), "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(MyArray<T, 3>), "");
    static_assert(!std::is_empty<Array>::value, "");
  }

  {
    using Array = std::array<T, 444>;
    static_assert(sizeof(Array) == sizeof(T) * 444, "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(T), "");
    static_assert(sizeof(Array) == sizeof(T[444]), "");
    static_assert(sizeof(Array) == sizeof(MyArray<T, 444>), "");
    static_assert(TEST_ALIGNOF(Array) == TEST_ALIGNOF(MyArray<T, 444>), "");
    static_assert(!std::is_empty<Array>::value, "");
  }
}

struct Empty {};

struct Aggregate {
  int i;
};

struct WithPadding {
  long double ld;
  char c;
};

#if TEST_STD_VER >= 11
struct alignas(TEST_ALIGNOF(std::max_align_t) * 2) Overaligned1 {};

struct alignas(TEST_ALIGNOF(std::max_align_t) * 2) Overaligned2 {
  char data[1000];
};

struct alignas(TEST_ALIGNOF(std::max_align_t)) Overaligned3 {
  char data[1000];
};

struct alignas(8) Overaligned4 {
  char c;
};

struct alignas(8) Overaligned5 {};
#endif

void test() {
  test_type<char>();
  test_type<short>();
  test_type<int>();
  test_type<long>();
  test_type<long long>();
  test_type<float>();
  test_type<double>();
  test_type<long double>();
  test_type<char[1]>();
  test_type<char[2]>();
  test_type<char[3]>();
  test_type<Empty>();
  test_type<Aggregate>();
  test_type<WithPadding>();

#if TEST_STD_VER >= 11
  test_type<std::max_align_t>();
  test_type<Overaligned1>();
  test_type<Overaligned2>();
  test_type<Overaligned3>();
  test_type<Overaligned4>();
  test_type<Overaligned5>();
#endif
}
