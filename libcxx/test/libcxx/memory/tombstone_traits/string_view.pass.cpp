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

// tombstone_traits<basic_string_view>

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <string_view>
#include <utility>

#include "test_macros.h"

// The tombstone relies on the low bit of the data pointer being unset, so characters must have an alignment of at
// least 2.
static_assert(!std::__has_tombstone_v<std::string_view>);
#ifndef TEST_HAS_NO_CHAR8_T
static_assert(!std::__has_tombstone_v<std::u8string_view>);
#endif
static_assert(std::__has_tombstone_v<std::u16string_view>);
static_assert(std::__has_tombstone_v<std::u32string_view>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::__has_tombstone_v<std::wstring_view>);
#endif

struct Empty {
  friend constexpr bool operator==(const Empty&, const Empty&) = default;
};

// Payloads that can fit into the available bytes given their padding

struct Size7Align1 {
  char payload[7] = {'h', 'e', 'l', 'l', 'o', '1', '\0'};

  friend constexpr bool operator==(const Size7Align1&, const Size7Align1&) = default;
};

struct Size6Align2 {
  short a = 1, b = 2, c = 3;

  friend constexpr bool operator==(const Size6Align2&, const Size6Align2&) = default;
};

struct Size4Align4 {
  int a = 1234;

  friend constexpr bool operator==(const Size4Align4&, const Size4Align4&) = default;
};

template <class View, class Payload>
constexpr void test_payload(View expected) {
  // start engaged, disengage
  {
    std::__tombstoned_value<View, Payload> tombstone(std::__init_engaged, expected);
    assert(sizeof(tombstone) == sizeof(View));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value().data() == expected.data());
    assert(tombstone.__get_value().size() == expected.size());
    assert(std::as_const(tombstone).__get_value() == expected);

    tombstone.__disengage(std::in_place, Payload{});
    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});
  }

  // start disengaged, then engage
  {
    std::__tombstoned_value<View, Payload> tombstone(std::__init_disengaged, Payload{});
    assert(sizeof(tombstone) == sizeof(View));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});

    tombstone.__engage(std::in_place, expected.data(), expected.size());
    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value().data() == expected.data());
    assert(tombstone.__get_value().size() == expected.size());
  }

  // a default constructed string_view has a null data pointer, but it's still engaged
  {
    std::__tombstoned_value<View, Payload> tombstone(std::__init_engaged);
    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value().data() == nullptr);
    assert(tombstone.__get_value().empty());
  }
}

template <class View>
constexpr void test_view(View expected) {
  test_payload<View, Empty>(expected);
  test_payload<View, Size7Align1>(expected);
  test_payload<View, Size6Align2>(expected);
  test_payload<View, Size4Align4>(expected);
}

constexpr bool test() {
  test_view<std::u16string_view>(u"hello");
  test_view<std::u32string_view>(U"hello");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_view<std::wstring_view>(L"hello");
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
