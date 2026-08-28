//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// template <class T, class S>
//   constexpr T shl(T x, S s) noexcept;

// Constraints: Each of T and S is a signed or unsigned integer type.

#include <bit>
#include <cassert>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

template <class T>
constexpr bool test() {
  using U             = std::make_unsigned_t<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  constexpr U highbit_u = U(1) << (width - 1);

  if constexpr (std::is_signed_v<T>) {
    ASSERT_SAME_TYPE(decltype(std::shl(T(), 0)), T);
    ASSERT_NOEXCEPT(std::shl(T(), 0));

    assert(std::shl(T(1), 0) == T(1));
    assert(std::shl(T(1), 1) == T(2));
    assert(std::shl(T(1), 2) == T(4));
    assert(std::shl(T(1), 3) == T(8));
    assert(std::shl(T(1), 4) == T(16));
    assert(std::shl(T(1), 5) == T(32));
    assert(std::shl(T(1), 6) == T(64));
    assert(std::shl(T(1), 7) == static_cast<T>(U(128)));

    // Overlong shifts return 0
    assert(std::shl(T(1), width) == T(0));
    assert(std::shl(T(1), width + 1) == T(0));
    assert(std::shl(T(1), width + 100) == T(0));
    assert(std::shl(T(1), std::numeric_limits<int>::max()) == T(0));

    assert(std::shl(T(-1), width) == T(0));
    assert(std::shl(T(-1), width + 5) == T(0));

    // Negative shift amounts shift right arithmetically (bidirectional)
    T highbit = static_cast<T>(highbit_u);
    assert(std::shl(highbit, -1) == (highbit >> 1));
    assert(std::shl(highbit, -2) == (highbit >> 2));
    assert(std::shl(highbit, -3) == (highbit >> 3));
    assert(std::shl(highbit, -4) == (highbit >> 4));
    assert(std::shl(highbit, -5) == (highbit >> 5));
    assert(std::shl(highbit, -6) == (highbit >> 6));
    assert(std::shl(highbit, -7) == (highbit >> 7));
    assert(std::shl(highbit, -(width - 1)) == (highbit >> (width - 1)));

    // Negative overlong shifts: sign-extended for negative values, 0 otherwise
    assert(std::shl(T(-1), -width) == T(-1));
    assert(std::shl(T(-1), -(width + 1)) == T(-1));
    assert(std::shl(T(-1), std::numeric_limits<int>::min()) == T(-1));
    assert(std::shl(T(42), -width) == T(0));
    assert(std::shl(T(42), -(width + 1)) == T(0));

    if constexpr (width == 128) {
      assert(std::shl(highbit, -64) == (highbit >> 64));
      assert(std::shl(highbit, -127) == (highbit >> 127));
      assert(std::shl(highbit, -128) == T(-1));
      assert(std::shl(highbit, -200) == T(-1));
    } else if constexpr (width == 256) {
      assert(std::shl(highbit, -64) == (highbit >> 64));
      assert(std::shl(highbit, -127) == (highbit >> 127));
      assert(std::shl(highbit, -128) == (highbit >> 128));
      assert(std::shl(highbit, -200) == (highbit >> 200));
      assert(std::shl(highbit, -255) == T(-1));
      assert(std::shl(highbit, -256) == T(-1));
    }
  } else {
    ASSERT_SAME_TYPE(decltype(std::shl(T(), 0)), T);
    ASSERT_NOEXCEPT(std::shl(T(), 0));

    assert(std::shl(T(1), 0) == T(1));
    assert(std::shl(T(1), 1) == T(2));
    assert(std::shl(T(1), 2) == T(4));
    assert(std::shl(T(1), 3) == T(8));
    assert(std::shl(T(1), 4) == T(16));
    assert(std::shl(T(1), 5) == T(32));
    assert(std::shl(T(1), 6) == T(64));
    assert(std::shl(T(1), 7) == T(128));

    // Overlong shifts return 0
    assert(std::shl(T(1), width) == T(0));
    assert(std::shl(T(1), width + 1) == T(0));
    assert(std::shl(T(1), width + 100) == T(0));
    assert(std::shl(T(1), std::numeric_limits<int>::max()) == T(0));

    assert(std::shl(T(~T(0)), width) == T(0));
    assert(std::shl(T(~T(0)), width + 5) == T(0));

    // Negative shift amounts shift right (bidirectional)
    T highbit = highbit_u;
    assert(std::shl(highbit, -1) == (highbit_u >> 1));
    assert(std::shl(highbit, -2) == (highbit_u >> 2));
    assert(std::shl(highbit, -3) == (highbit_u >> 3));
    assert(std::shl(highbit, -4) == (highbit_u >> 4));
    assert(std::shl(highbit, -5) == (highbit_u >> 5));
    assert(std::shl(highbit, -6) == (highbit_u >> 6));
    assert(std::shl(highbit, -7) == (highbit_u >> 7));
    assert(std::shl(highbit, -(width - 1)) == T(1));

    // Negative overlong shifts: 0 for unsigned
    assert(std::shl(T(~T(0)), -width) == T(0));
    assert(std::shl(T(~T(0)), -(width + 1)) == T(0));
    assert(std::shl(T(1), std::numeric_limits<int>::min()) == T(0));

    if constexpr (width == 128) {
      assert(std::shl(T(1), 64) == T(1) << 64);
      assert(std::shl(T(1), 127) == T(1) << 127);
      assert(std::shl(T(1), 128) == T(0));
      assert(std::shl(T(1), 200) == T(0));
      assert(std::shl(T(1) << 127, -1) == T(1) << 126);
      assert(std::shl(T(1) << 127, -127) == T(1));
      assert(std::shl(T(1) << 127, -128) == T(0));
      assert(std::shl(T(1) << 127, -200) == T(0));
    } else if constexpr (width == 256) {
      assert(std::shl(T(1), 64) == T(1) << 64);
      assert(std::shl(T(1), 127) == T(1) << 127);
      assert(std::shl(T(1), 128) == T(1) << 128);
      assert(std::shl(T(1), 200) == T(1) << 200);
      assert(std::shl(T(1) << 127, -1) == T(1) << 126);
      assert(std::shl(T(1) << 127, -127) == T(1));
      assert(std::shl(T(1) << 127, -128) == T(0));
      assert(std::shl(T(1) << 127, -200) == T(0));
      assert(std::shl(T(1), 256) == T(0));
      assert(std::shl(T(1), 300) == T(0));
    }
  }

  return true;
}

struct A {};
enum E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <class T>
concept can_shl_value = requires(T value) { std::shl(value, 1); };

template <class T>
concept can_shl_shift = requires(T shift) { std::shl(1, shift); };

int main(int, char**) {
  static_assert(!can_shl_value<bool>);
  static_assert(!can_shl_value<char>);
  static_assert(!can_shl_value<wchar_t>);
  static_assert(!can_shl_value<char8_t>);
  static_assert(!can_shl_value<char16_t>);
  static_assert(!can_shl_value<char32_t>);
  static_assert(!can_shl_value<A>);
  static_assert(!can_shl_value<A*>);
  static_assert(!can_shl_value<E1>);
  static_assert(!can_shl_value<E2>);

  static_assert(!can_shl_shift<bool>);
  static_assert(!can_shl_shift<char>);
  static_assert(!can_shl_shift<wchar_t>);
  static_assert(!can_shl_shift<char8_t>);
  static_assert(!can_shl_shift<char16_t>);
  static_assert(!can_shl_shift<char32_t>);
  static_assert(!can_shl_shift<A>);
  static_assert(!can_shl_shift<A*>);
  static_assert(!can_shl_shift<E1>);
  static_assert(!can_shl_shift<E2>);

  constexpr auto test_type = []<class T> {
    static_assert(test<T>());
    test<T>();
  };

  types::for_each(types::unsigned_integer_types{}, test_type);
  types::for_each(types::signed_integer_types{}, test_type);

#if TEST_HAS_BITINT
  using bitint_types = types::type_list<
      signed _BitInt(32),
      unsigned _BitInt(32),
      signed _BitInt(64),
      unsigned _BitInt(64)
#  if __BITINT_MAXWIDTH__ >= 128
          ,
      signed _BitInt(128),
      unsigned _BitInt(128)
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
          ,
      signed _BitInt(256),
      unsigned _BitInt(256)
#  endif
      >;
  types::for_each(bitint_types{}, test_type);
#endif

  return 0;
}
