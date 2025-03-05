//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <atomic>

// template <class T>
// struct atomic_ref
// {
//    using value_type = T;
//    using difference_type = value_type;      // only for atomic_ref<Integral> and
//                                             // atomic_ref<Floating> specializations
//    using difference_type = std::ptrdiff_t;  // only for atomic_ref<T*> specializations
//
//    explicit atomic_ref(T&);
//    atomic_ref(const atomic_ref&) noexcept;
//    atomic_ref& operator=(const atomic_ref&) = delete;
// };

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept has_difference_type = requires { typename T::difference_type; };

template <class T>
void check_member_types() {
  if constexpr ((std::is_integral_v<T> && !std::is_same_v<T, bool>) || std::is_floating_point_v<T>) {
    ASSERT_SAME_TYPE(typename std::atomic_ref<T>::value_type, T);
    ASSERT_SAME_TYPE(typename std::atomic_ref<T>::difference_type, T);
  } else if constexpr (std::is_pointer_v<T>) {
    ASSERT_SAME_TYPE(typename std::atomic_ref<T>::value_type, T);
    ASSERT_SAME_TYPE(typename std::atomic_ref<T>::difference_type, std::ptrdiff_t);
  } else {
    ASSERT_SAME_TYPE(typename std::atomic_ref<T>::value_type, T);
    static_assert(!has_difference_type<std::atomic_ref<T>>);
  }
}

template <class T>
void test() {
  // value_type and difference_type (except for primary template)
  check_member_types<T>();

  static_assert(std::is_nothrow_copy_constructible_v<std::atomic_ref<T>>);

  static_assert(!std::is_copy_assignable_v<std::atomic_ref<T>>);

  // explicit constructor
  static_assert(!std::is_convertible_v<T, std::atomic_ref<T>>);
  static_assert(std::is_constructible_v<std::atomic_ref<T>, T&>);
}

void testall() {
  // Primary template
  struct Empty {};
  test<Empty>();
  struct Trivial {
    int a;
    float b;
  };
  test<Trivial>();
  test<bool>();

  // Partial specialization for pointer types
  test<void*>();

  // Specialization for integral types
  // + character types
  test<char>();
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();
  test<wchar_t>();
  // + standard signed integer types
  test<signed char>();
  test<short>();
  test<int>();
  test<long>();
  test<long long>();
  // + standard unsigned integer types
  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
  // + any other types needed by the typedefs in the header <cstdint>
  test<std::int8_t>();
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();
  test<std::int_fast8_t>();
  test<std::int_fast16_t>();
  test<std::int_fast32_t>();
  test<std::int_fast64_t>();
  test<std::int_least8_t>();
  test<std::int_least16_t>();
  test<std::int_least32_t>();
  test<std::int_least64_t>();
  test<std::intmax_t>();
  test<std::intptr_t>();
  test<std::uint8_t>();
  test<std::uint16_t>();
  test<std::uint32_t>();
  test<std::uint64_t>();
  test<std::uint_fast8_t>();
  test<std::uint_fast16_t>();
  test<std::uint_fast32_t>();
  test<std::uint_fast64_t>();
  test<std::uint_least8_t>();
  test<std::uint_least16_t>();
  test<std::uint_least32_t>();
  test<std::uint_least64_t>();
  test<std::uintmax_t>();
  test<std::uintptr_t>();

  // Specialization for floating-point types
  // + floating-point types
  test<float>();
  test<double>();
  test<long double>();
  // + TODO extended floating-point types
}
