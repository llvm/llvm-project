//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

#include "test_macros.h"

template <class T>
concept has_byteswap = requires(T t) {
  std::byteswap(t);
};

static_assert(!has_byteswap<void*>);
static_assert(!has_byteswap<float>);
static_assert(!has_byteswap<char[2]>);
static_assert(!has_byteswap<std::byte>);

template <class T>
constexpr void test_num(T in, T expected) {
  assert(std::byteswap(in) == expected);
  ASSERT_SAME_TYPE(decltype(std::byteswap(in)), decltype(in));
  ASSERT_NOEXCEPT(std::byteswap(in));
}

template <class T>
constexpr std::pair<T, T> get_test_data() {
  switch (sizeof(T)) {
  case 2:
    return {static_cast<T>(0x1234), static_cast<T>(0x3412)};
  case 4:
    return {static_cast<T>(0x60AF8503), static_cast<T>(0x0385AF60)};
  case 8:
    return {static_cast<T>(0xABCDFE9477936406), static_cast<T>(0x0664937794FECDAB)};
  default:
    assert(false);
    return {}; // for MSVC, whose `assert` is tragically not [[noreturn]]
  }
}

template <class T>
constexpr void test_implementation_defined_size() {
  const auto [in, expected] = get_test_data<T>();
  test_num<T>(in, expected);
}

constexpr bool test() {
  test_num<uint8_t>(0xAB, 0xAB);
  test_num<uint16_t>(0xCDEF, 0xEFCD);
  test_num<uint32_t>(0x01234567, 0x67452301);
  test_num<uint64_t>(0x0123456789ABCDEF, 0xEFCDAB8967452301);

  test_num<int8_t>(static_cast<int8_t>(0xAB), static_cast<int8_t>(0xAB));
  test_num<int16_t>(static_cast<int16_t>(0xCDEF), static_cast<int16_t>(0xEFCD));
  test_num<int32_t>(0x01234567, 0x67452301);
  test_num<int64_t>(0x0123456789ABCDEF, 0xEFCDAB8967452301);

#ifndef TEST_HAS_NO_INT128
  const auto in = static_cast<__uint128_t>(0x0123456789ABCDEF) << 64 | 0x13579BDF02468ACE;
  const auto expected = static_cast<__uint128_t>(0xCE8A4602DF9B5713) << 64 | 0xEFCDAB8967452301;
  test_num<__uint128_t>(in, expected);
  test_num<__int128_t>(in, expected);
#endif

  test_num<bool>(true, true);
  test_num<bool>(false, false);
  test_num<char>(static_cast<char>(0xCD), static_cast<char>(0xCD));
  test_num<unsigned char>(0xEF, 0xEF);
  test_num<signed char>(0x45, 0x45);
  test_num<char8_t>(0xAB, 0xAB);
  test_num<char16_t>(0xABCD, 0xCDAB);
  test_num<char32_t>(0xABCDEF01, 0x01EFCDAB);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_implementation_defined_size<wchar_t>();
#endif

  test_implementation_defined_size<short>();
  test_implementation_defined_size<unsigned short>();
  test_implementation_defined_size<int>();
  test_implementation_defined_size<unsigned int>();
  test_implementation_defined_size<long>();
  test_implementation_defined_size<unsigned long>();
  test_implementation_defined_size<long long>();
  test_implementation_defined_size<unsigned long long>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
