//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// tombstone_traits<reference_wrapper>

#include <__memory/tombstone_traits.h>
#include <cassert>
#include <functional>
#include <utility>

struct Empty {};

struct Incomplete;
static_assert(std::__has_tombstone_v<std::reference_wrapper<Incomplete>>);

template <class T>
constexpr void test_type(T& value) {
  // start engaged, disengage
  {
    std::__tombstoned_value<std::reference_wrapper<T>, Empty> tombstone(std::__init_engaged, value);
    assert(sizeof(tombstone) == sizeof(std::reference_wrapper<T>));

    assert(tombstone.__is_engaged());
    assert(&tombstone.__get_value().get() == &value);
    assert(&std::as_const(tombstone).__get_value().get() == &value);
    assert(&std::move(tombstone).__get_value().get() == &value);

    tombstone.__disengage(std::in_place);
    assert(!tombstone.__is_engaged());
  }

  // start disengaged, then engage
  {
    std::__tombstoned_value<std::reference_wrapper<T>, Empty> tombstone(std::__init_disengaged);
    assert(sizeof(tombstone) == sizeof(std::reference_wrapper<T>));

    assert(!tombstone.__is_engaged());

    tombstone.__engage(std::in_place, value);
    assert(tombstone.__is_engaged());
    assert(&tombstone.__get_value().get() == &value);
  }
}

struct UserDefined {
  int a = 1;
};

constexpr bool test() {
  int my_int = 1;
  test_type<int>(my_int);
  test_type<const int>(my_int);

  char my_char = 'a';
  test_type<char>(my_char);

  UserDefined my_struct;
  test_type<UserDefined>(my_struct);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
