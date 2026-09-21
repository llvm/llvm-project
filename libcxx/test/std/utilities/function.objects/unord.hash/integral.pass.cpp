//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

#include <functional>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "constexpr_hash.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
#  include "poisoned_hash_helper.h"
#endif

template <class T, template <class> class THash = std::hash >
TEST_CONSTEXPR_CXX26 void test() {
#if TEST_STD_VER >= 11
  test_hash_disabled<const T, THash<const T>>();
  test_hash_disabled<volatile T, THash<volatile T>>();
  test_hash_disabled<const volatile T, THash<const volatile T>>();
#endif

  // typedef std::hash<T> H;
  typedef THash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "");
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));
    H h;

    for (int i = 0; i <= 5; ++i)
    {
        T t(static_cast<T>(i));
        const bool small = std::integral_constant<bool, sizeof(T) <= sizeof(std::size_t)>::value; // avoid compiler warnings
        if (small)
        {
            const std::size_t result = h(t);
            LIBCPP_ASSERT(result == static_cast<std::size_t>(t));
            ((void)result); // Prevent unused warning
        }
    }
}

template <template <typename> typename THash >
TEST_CONSTEXPR_CXX26 bool test_with_hash() {
  test<bool, THash>();
  test<char, THash>();
  test<signed char, THash>();
  test<unsigned char, THash>();
  test<char16_t, THash>();
  test<char32_t, THash>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, THash>();
#endif
  test<short, THash>();
  test<unsigned short, THash>();
  test<int, THash>();
  test<unsigned int, THash>();
  test<long, THash>();
  test<unsigned long, THash>();
  test<long long, THash>();
  test<unsigned long long, THash>();

  //  LWG #2119
  test<std::ptrdiff_t, THash>();
  test<std::size_t, THash>();

  test<std::int8_t, THash>();
  test<std::int16_t, THash>();
  test<std::int32_t, THash>();
  test<std::int64_t, THash>();

  test<std::int_fast8_t, THash>();
  test<std::int_fast16_t, THash>();
  test<std::int_fast32_t, THash>();
  test<std::int_fast64_t, THash>();

  test<std::int_least8_t, THash>();
  test<std::int_least16_t, THash>();
  test<std::int_least32_t, THash>();
  test<std::int_least64_t, THash>();

  test<std::intmax_t, THash>();
  test<std::intptr_t, THash>();

  test<std::uint8_t, THash>();
  test<std::uint16_t, THash>();
  test<std::uint32_t, THash>();
  test<std::uint64_t, THash>();

  test<std::uint_fast8_t, THash>();
  test<std::uint_fast16_t, THash>();
  test<std::uint_fast32_t, THash>();
  test<std::uint_fast64_t, THash>();

  test<std::uint_least8_t, THash>();
  test<std::uint_least16_t, THash>();
  test<std::uint_least32_t, THash>();
  test<std::uint_least64_t, THash>();

  test<std::uintmax_t, THash>();
  test<std::uintptr_t, THash>();

#ifndef TEST_HAS_NO_INT128
  test<__int128_t, THash>();
  test<__uint128_t, THash>();
#endif

  return true;
}

int main(int, char**) {
  assert(test_with_hash<std::hash>());

#if TEST_STD_VER >= 26
  static_assert(test_with_hash<support::constexpr_hash>());
#endif

  return 0;
}
