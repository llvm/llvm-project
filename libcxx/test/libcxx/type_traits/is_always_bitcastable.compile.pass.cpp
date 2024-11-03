//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <type_traits>
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
//
// __is_always_bitcastable<_From, _To>

#include "test_macros.h"
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__type_traits/is_always_bitcastable.h>

#include <climits>
#include <cstdint>
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#include <cwchar>
#endif
#include "type_algorithms.h"

// To test pointers to functions.
void Func1() {}
using FuncPtr1 = decltype(&Func1);
int Func2() { return 0; }
using FuncPtr2 = decltype(&Func2);

template <bool Expected, class T, class U>
constexpr void check_one() {
  static_assert(std::__is_always_bitcastable<T, U>::value == Expected);
}

template <bool Expected, class T, class U>
constexpr void check_with_volatile() {
  check_one<Expected, T, U>();
  check_one<Expected, volatile T, U>();
  check_one<Expected, T, volatile U>();
  check_one<Expected, volatile T, volatile U>();
}

template <bool Expected, class T, class U>
constexpr void check_with_cv() {
  check_with_volatile<Expected, T, U>();
  check_with_volatile<Expected, const T, U>();
  check_with_volatile<Expected, T, const U>();
  check_with_volatile<Expected, const T, const U>();
}

template <bool Expected, class Types1, class Types2 = Types1>
constexpr void check() {
  meta::for_each(Types1{}, []<class T>() {
    meta::for_each(Types2{}, []<class U>() {
      check_with_cv<Expected, T, U>();
    });
  });
}

template <bool Expected, class Types1, class Types2>
constexpr void check_both_ways() {
  check<Expected, Types1, Types2>();
  check<Expected, Types2, Types1>();
}

constexpr void test() {
  // Arithmetic types.
  {
    // Bit-castable arithmetic types.

    // 8-bit types.
    using integral_8 = meta::type_list<char8_t, int8_t, uint8_t>;
    using chars = meta::type_list<char, unsigned char, signed char>;
#if CHAR_BIT == 8
    check<true, meta::concatenate_t<integral_8, chars>>();
#else
    check<true, integral_8>();
    check<true, chars>();
#endif

    // 16-bit types.
    using integral_16 = meta::type_list<char16_t, int16_t, uint16_t>;
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && __WCHAR_WIDTH__ == 16
    check<true, meta::concatenate_t<integral_16, meta::type_list<wchar_t>>>();
#else
    check<true, integral_16>();
#endif

    // 32-bit types.
    using integral_32 = meta::type_list<char32_t, int32_t, uint32_t>;
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && __WCHAR_WIDTH__ == 32
    check<true, meta::concatenate_t<integral_32, meta::type_list<wchar_t>>>();
#else
    check<true, integral_32>();
#endif

    // 64-bit types.
    using integral_64 = meta::type_list<int64_t, uint64_t>;
    check<true, integral_64>();

    // 128-bit types.
#ifndef TEST_HAS_NO_INT128
    check<true, meta::type_list<__int128_t, __uint128_t>>();
#endif

    // Bool.
    check<true, meta::type_list<bool>, meta::concatenate_t<meta::type_list<bool>, integral_8>>();

    // Non-bit-castable arithmetic types.

    // Floating-point.
    check_both_ways<false, meta::floating_point_types, meta::integral_types>();
    check_both_ways<false, meta::type_list<float>, meta::type_list<double, long double>>();
    check_both_ways<false, meta::type_list<double>, meta::type_list<float, long double>>();
    check_both_ways<false, meta::type_list<long double>, meta::type_list<float, double>>();

    // Different sizes.
    check_both_ways<false, integral_8, meta::concatenate_t<integral_16, integral_32, integral_64>>();
    check_both_ways<false, integral_16, meta::concatenate_t<integral_8, integral_32, integral_64>>();
    check_both_ways<false, integral_32, meta::concatenate_t<integral_8, integral_16, integral_64>>();
    check_both_ways<false, integral_64, meta::concatenate_t<integral_8, integral_16, integral_32>>();

    // Different representations -- can convert from bool to other integral types, but not vice versa.
    check<true, meta::type_list<bool>, integral_8>();
    using larger_than_bool = meta::concatenate_t<
      integral_16,
      integral_32,
      integral_64,
      meta::floating_point_types>;
    check<false, meta::type_list<bool>, larger_than_bool>();
    check<false, meta::concatenate_t<integral_8, larger_than_bool>, meta::type_list<bool>>();

    // Different representations -- floating point vs. integral.
    check_both_ways<false, meta::floating_point_types, meta::integral_types>();
  }

  // Enumerations.
  {
    enum E1 { Value1 };
    enum E2 { Value2 };
    check<true, meta::type_list<E1>>();
    check_both_ways<false, meta::type_list<E1>, meta::type_list<E2>>();

    enum class ScopedE1 { Value1 };
    enum class ScopedE2 { Value1 };
    check<true, meta::type_list<ScopedE1>>();
    check_both_ways<false, meta::type_list<ScopedE1>, meta::type_list<ScopedE2>>();
  }

  // Pointers.
  {
    check<true, meta::type_list<int*>>();
    check_both_ways<false, meta::type_list<int*>, meta::type_list<const int*, long*, void*>>();

    check<true, meta::type_list<FuncPtr1>>();
    check_both_ways<false, meta::type_list<FuncPtr1>, meta::type_list<FuncPtr2>>();
  }

  // Pointers to members.
  {
    struct S {
      int mem_obj1 = 0;
      long mem_obj2 = 0;
      void MemFunc1() {}
      int MemFunc2() { return 0; }
    };
    using MemObjPtr1 = decltype(&S::mem_obj1);
    using MemObjPtr2 = decltype(&S::mem_obj2);
    using MemFuncPtr1 = decltype(&S::MemFunc1);
    using MemFuncPtr2 = decltype(&S::MemFunc2);

    check<true, meta::type_list<MemObjPtr1>>();
    check<true, meta::type_list<MemFuncPtr1>>();
    check_both_ways<false, meta::type_list<MemObjPtr1>, meta::type_list<MemObjPtr2>>();
    check_both_ways<false, meta::type_list<MemFuncPtr1>, meta::type_list<MemFuncPtr2>>();
  }

  // Trivial classes.
  {
    struct S1 {};
    check<true, meta::type_list<S1>>();

    struct S2 {};
    check_both_ways<false, meta::type_list<S1>, meta::type_list<S2>>();

    // Having a `volatile` member doesn't prevent a class type from being considered trivially copyable. This is
    // unfortunate behavior but it is consistent with the Standard.
    struct VolatileMembersS {
      volatile int x;
    };
    check<true, meta::type_list<VolatileMembersS>>();
  }

  // Trivial unions.
  {
    union U1 {};
    check<true, meta::type_list<U1>>();

    union U2 {};
    check_both_ways<false, meta::type_list<U1>, meta::type_list<U2>>();

    union VolatileMembersU {
      volatile int x;
    };
    check<true, meta::type_list<VolatileMembersU>>();
  }

  // References are not objects, and thus are not bit-castable.
  {
    check_both_ways<false, meta::type_list<int&>, meta::type_list<int&>>();
  }

  // Arrays.
  {
    check<true, meta::type_list<int[8]>>();
  }
}
