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

// tombstone_traits<unique_ptr>

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <memory>
#include <utility>

struct UserDefined {};

struct CustomDeleter {
  constexpr void operator()(int* ptr) const { delete ptr; }
};

// Only pointers that have a tombstone themselves and the default deleter get a tombstone
static_assert(std::__has_tombstone_v<std::unique_ptr<int>>);
static_assert(std::__has_tombstone_v<std::unique_ptr<int[]>>);
static_assert(std::__has_tombstone_v<std::unique_ptr<int*>>);
static_assert(!std::__has_tombstone_v<std::unique_ptr<char>>);
static_assert(!std::__has_tombstone_v<std::unique_ptr<char[]>>);
static_assert(!std::__has_tombstone_v<std::unique_ptr<UserDefined>>);
static_assert(!std::__has_tombstone_v<std::unique_ptr<int, CustomDeleter>>);

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

template <class Payload>
constexpr void test_single_object() {
  // start engaged, disengage
  {
    std::__tombstoned_value<std::unique_ptr<int>, Payload> tombstone(std::__init_engaged, new int(3));
    assert(sizeof(tombstone) == sizeof(std::unique_ptr<int>));

    assert(tombstone.__is_engaged());
    assert(*tombstone.__get_value() == 3);
    assert(*std::as_const(tombstone).__get_value() == 3);

    tombstone.__disengage(std::in_place, Payload{});
    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});
  }

  // start disengaged, then engage
  {
    std::__tombstoned_value<std::unique_ptr<int>, Payload> tombstone(std::__init_disengaged, Payload{});
    assert(sizeof(tombstone) == sizeof(std::unique_ptr<int>));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});

    tombstone.__engage(std::in_place, new int(3));
    assert(tombstone.__is_engaged());
    assert(*tombstone.__get_value() == 3);
  }

  // an empty unique_ptr is still engaged
  {
    std::__tombstoned_value<std::unique_ptr<int>, Payload> tombstone(std::__init_engaged);
    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == nullptr);
  }

  // moving the value out leaves it engaged with an empty unique_ptr
  {
    std::__tombstoned_value<std::unique_ptr<int>, Payload> tombstone(std::__init_engaged, new int(3));
    std::unique_ptr<int> moved = std::move(tombstone).__get_value();
    assert(*moved == 3);

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == nullptr);
  }
}

template <class Payload>
constexpr void test_array() {
  // start engaged, disengage
  {
    std::__tombstoned_value<std::unique_ptr<int[]>, Payload> tombstone(std::__init_engaged, new int[2]{3, 4});
    assert(sizeof(tombstone) == sizeof(std::unique_ptr<int[]>));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value()[0] == 3);
    assert(tombstone.__get_value()[1] == 4);

    tombstone.__disengage(std::in_place, Payload{});
    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});
  }

  // start disengaged, then engage
  {
    std::__tombstoned_value<std::unique_ptr<int[]>, Payload> tombstone(std::__init_disengaged, Payload{});
    assert(sizeof(tombstone) == sizeof(std::unique_ptr<int[]>));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload() == Payload{});

    tombstone.__engage(std::in_place, new int[2]{3, 4});
    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value()[0] == 3);
    assert(tombstone.__get_value()[1] == 4);
  }
}

constexpr bool test() {
  test_single_object<Size7Align1>();
  test_single_object<Size6Align2>();
  test_single_object<Size4Align4>();

  test_array<Size7Align1>();
  test_array<Size6Align2>();
  test_array<Size4Align4>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
