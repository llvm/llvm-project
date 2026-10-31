//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRED: std-at-least-c++20

// UNSUPPORTED: 32-bit-pointer
// TODO: Expand tests to accommodate 32-bit pointers

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <string_view>
#include <utility>

// Payloads that can fit into the available bytes given their padding

struct Size7Align1 {
  char payload[7] = {
      'h',
      'e',
      'l',
      'l',
      'o',
      '1',
      '\0',
  };

  constexpr std::string_view getPayload() & { return "ref"; }
  constexpr std::string_view getPayload() && { return "move"; }
  constexpr std::string_view getPayload() const& { return "const ref"; }
  constexpr std::string_view getPayload() const&& { return "const move"; }
};

struct Size6Align2 {
  short a = 1, b = 2, c = 3;
};

struct Size4Align4 {
  int a = 1234;
};

constexpr bool test() {
  // start engaged, disengage
  {
    int my_int = 1;
    std::__tombstoned_value<int*, Size7Align1> tombstone(std::__init_engaged, &my_int);
    assert(sizeof(tombstone) == sizeof(&my_int));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == &my_int);
    assert(std::as_const(tombstone.__get_value()) == &my_int);
    assert(std::move(tombstone.__get_value()) == &my_int);
    assert(std::move(std::as_const(tombstone)).__get_value() == &my_int);
    assert(*tombstone.__get_value() == my_int);

    tombstone.__disengage(std::in_place, Size7Align1{});
    assert(!tombstone.__is_engaged());

    assert(std::string_view(tombstone.__get_payload().payload) == "hello1");
    assert(tombstone.__get_payload().getPayload() == "ref");
    assert(std::as_const(tombstone.__get_payload()).getPayload() == "const ref");
    assert(std::move(tombstone.__get_payload()).getPayload() == "move");
    assert(std::move(std::as_const(tombstone.__get_payload())).getPayload() == "const move");
  }

  {
    short my_short = 1;
    std::__tombstoned_value<short*, Size6Align2> tombstone(std::__init_engaged, &my_short);
    assert(sizeof(tombstone) == sizeof(&my_short));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == &my_short);
    assert(*tombstone.__get_value() == my_short);

    tombstone.__disengage(std::in_place, Size6Align2{});
    assert(!tombstone.__is_engaged());

    assert(tombstone.__get_payload().a == 1);
    assert(tombstone.__get_payload().b == 2);
    assert(tombstone.__get_payload().c == 3);
  }

  {
    long my_long = 3;
    std::__tombstoned_value<long*, Size4Align4> tombstone(std::__init_engaged, &my_long);
    assert(sizeof(tombstone) == sizeof(&my_long));

    assert(tombstone.__is_engaged());
    assert(tombstone.__get_value() == &my_long);
    assert(*tombstone.__get_value() == my_long);

    tombstone.__disengage(std::in_place, Size4Align4{});
    assert(!tombstone.__is_engaged());

    assert(tombstone.__get_payload().a == 1234);
  }

  // start disengaged, then engage
  {
    int my_int = 1;
    std::__tombstoned_value<int*, Size7Align1> tombstone(std::__init_disengaged, Size7Align1{});
    assert(sizeof(tombstone) == sizeof(&my_int));

    assert(!tombstone.__is_engaged());
    assert(std::string_view(tombstone.__get_payload().payload) == "hello1");

    tombstone.__engage(std::in_place, &my_int);
    assert(tombstone.__is_engaged());

    assert(tombstone.__get_value() == &my_int);
    assert(*tombstone.__get_value() == my_int);
  }

  {
    short my_short = 1;
    std::__tombstoned_value<short*, Size6Align2> tombstone(std::__init_disengaged, Size6Align2{});
    assert(sizeof(tombstone) == sizeof(&my_short));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload().a == 1);
    assert(tombstone.__get_payload().b == 2);
    assert(tombstone.__get_payload().c == 3);

    tombstone.__engage(std::in_place, &my_short);
    assert(tombstone.__is_engaged());

    assert(tombstone.__get_value() == &my_short);
    assert(*tombstone.__get_value() == my_short);
  }

  {
    long my_long = 3;
    std::__tombstoned_value<long*, Size4Align4> tombstone(std::__init_disengaged, Size4Align4{});
    assert(sizeof(tombstone) == sizeof(&my_long));

    assert(!tombstone.__is_engaged());
    assert(tombstone.__get_payload().a == 1234);

    tombstone.__engage(std::in_place, &my_long);
    assert(tombstone.__is_engaged());

    assert(tombstone.__get_value() == &my_long);
    assert(*tombstone.__get_value() == my_long);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
