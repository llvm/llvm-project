//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// Test nested types

// template <class T>
// class atomic
// {
// public:
//     typedef T value_type;
// };

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include <thread>
#endif

// detect existence of the difference_type member type
template <class...>
using myvoid_t = void;
template <typename T, typename = void>
struct has_difference_type : std::false_type {};
template <typename T>
struct has_difference_type<T, myvoid_t<typename T::difference_type> > : std::true_type {};

template <class T,
          bool Integral = (std::is_integral<T>::value && !std::is_same<T, bool>::value),
          bool Floating = std::is_floating_point<T>::value,
          bool Pointer  = std::is_pointer<T>::value>
struct test_atomic {
  test_atomic() {
    static_assert(!Integral && !Floating && !Pointer, "");
    using A = std::atomic<T>;
    A a;
    (void)a;
    static_assert(std::is_same<typename A::value_type, T>::value, "");
    static_assert(!has_difference_type<A>::value, "");
  }
};

template <class T>
struct test_atomic<T, /*Integral=*/true, false, false> {
  test_atomic() {
    static_assert(!std::is_same<T, bool>::value, "");
    using A = std::atomic<T>;
    A a;
    (void)a;
    static_assert(std::is_same<typename A::value_type, T>::value, "");
    static_assert(std::is_same<typename A::difference_type, T>::value, "");
  }
};

template <class T>
struct test_atomic<T, false, /*Floating=*/true, false> {
  test_atomic() {
    using A = std::atomic<T>;
    A a;
    (void)a;
    static_assert(std::is_same<typename A::value_type, T>::value, "");
#if TEST_STD_VER >= 20
    static_assert(std::is_same<typename A::difference_type, T>::value, "");
#else
    static_assert(!has_difference_type<A>::value, "");
#endif
  }
};

template <class T>
struct test_atomic<T, false, false, /*Pointer=*/true> {
  test_atomic() {
    using A = std::atomic<T>;
    A a;
    (void)a;
    static_assert(std::is_same<typename A::value_type, T>::value, "");
    static_assert(std::is_same<typename A::difference_type, std::ptrdiff_t>::value, "");
  }
};

template <class T>
void test() {
  using A = std::atomic<T>;
  static_assert(std::is_same<typename A::value_type, T>::value, "");
  test_atomic<T>();
}

struct TriviallyCopyable {
  int i_;
};

struct WeirdTriviallyCopyable {
  char i, j, k; /* the 3 chars of doom */
};

struct PaddedTriviallyCopyable {
  char i;
  int j; /* probably lock-free? */
};

struct LargeTriviallyCopyable {
  int i, j[127]; /* decidedly not lock-free */
};

int main(int, char**) {
  test<bool>();
  test<char>();
  test<signed char>();
  test<unsigned char>();
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<long>();
  test<unsigned long>();
  test<long long>();
  test<unsigned long long>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  test<std::int_least8_t>();
  test<std::uint_least8_t>();
  test<std::int_least16_t>();
  test<std::uint_least16_t>();
  test<std::int_least32_t>();
  test<std::uint_least32_t>();
  test<std::int_least64_t>();
  test<std::uint_least64_t>();

  test<std::int_fast8_t>();
  test<std::uint_fast8_t>();
  test<std::int_fast16_t>();
  test<std::uint_fast16_t>();
  test<std::int_fast32_t>();
  test<std::uint_fast32_t>();
  test<std::int_fast64_t>();
  test<std::uint_fast64_t>();

  test< std::int8_t>();
  test<std::uint8_t>();
  test< std::int16_t>();
  test<std::uint16_t>();
  test< std::int32_t>();
  test<std::uint32_t>();
  test< std::int64_t>();
  test<std::uint64_t>();

  test<std::intptr_t>();
  test<std::uintptr_t>();
  test<std::size_t>();
  test<std::ptrdiff_t>();
  test<std::intmax_t>();
  test<std::uintmax_t>();

  test<std::uintmax_t>();
  test<std::uintmax_t>();

  test<void (*)(int)>();
  test<void*>();
  test<const void*>();
  test<int*>();
  test<const int*>();

  test<TriviallyCopyable>();
  test<PaddedTriviallyCopyable>();
#ifndef __APPLE__ // Apple doesn't ship libatomic
  /*
        These aren't going to be lock-free,
        so some libatomic.a is necessary.
    */
  test<WeirdTriviallyCopyable>();
  test<LargeTriviallyCopyable>();
#endif

#ifndef TEST_HAS_NO_THREADS
  test<std::thread::id>();
#endif
  test<std::chrono::nanoseconds>();
  test<float>();

#if TEST_STD_VER >= 20
  test<std::atomic_signed_lock_free::value_type>();
  static_assert(std::is_signed_v<std::atomic_signed_lock_free::value_type>);
  static_assert(std::is_integral_v<std::atomic_signed_lock_free::value_type>);

  test<std::atomic_unsigned_lock_free::value_type>();
  static_assert(std::is_unsigned_v<std::atomic_unsigned_lock_free::value_type>);
  static_assert(std::is_integral_v<std::atomic_unsigned_lock_free::value_type>);
/*
    test<std::shared_ptr<int>>();
*/
#endif

  return 0;
}
