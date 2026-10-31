//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// UNSUPPORTED: 32-bit-pointer
// TODO: Expand tests to accommodate 32-bit pointers

// tombstone_traits<basic_string>

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

struct Empty {
  friend constexpr bool operator==(const Empty&, const Empty&) = default;
};

struct Size7Align1 {
  char payload[7] = {'h', 'e', 'l', 'l', 'o', '1', '\0'};

  friend constexpr bool operator==(const Size7Align1&, const Size7Align1&) = default;
};

struct Size16Align8 {
  long long a = 1, b = 2;

  friend constexpr bool operator==(const Size16Align8&, const Size16Align8&) = default;
};

template <class String, class Payload>
constexpr void test_payload(std::size_t size) {
  using CharT = typename String::value_type;
  const String expected(size, CharT('a'));

  // start engaged, disengage
  {
    std::__tombstoned_value<String, Payload> tombstone(std::__init_engaged, expected);
    assert(sizeof(tombstone) == sizeof(String));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == expected);
    assert(std::as_const(tombstone).__get_value() == expected);

    tombstone.__disengage(std::in_place, Payload{});
    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});
  }

  // start disengaged, then engage
  {
    std::__tombstoned_value<String, Payload> tombstone(std::__init_disengaged, Payload{});
    assert(sizeof(tombstone) == sizeof(String));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});

    tombstone.__engage(std::in_place, expected);
    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == expected);
  }
}

template <class String>
constexpr void test_string() {
  // Exercises both the short and the long representation
  for (std::size_t size : {std::size_t(0), std::size_t(1), std::size_t(100)}) {
    test_payload<String, Empty>(size);
    test_payload<String, Size7Align1>(size);
    test_payload<String, Size16Align8>(size);
  }
}

constexpr bool test() {
  test_string<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_string<std::wstring>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test_string<std::u8string>();
#endif
  test_string<std::u16string>();
  test_string<std::u32string>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
