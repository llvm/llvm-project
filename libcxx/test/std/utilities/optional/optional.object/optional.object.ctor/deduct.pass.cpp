//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <optional>

// template<class T>
//   optional(T) -> optional<T>;

#include <cassert>
#include <optional>

#include "test_macros.h"

struct A {
  friend constexpr bool operator==(const A&, const A&) { return true; }
};

template <typename T>
constexpr void test_deduct(T arg) {
  std::optional opt(arg);

  ASSERT_SAME_TYPE(decltype(opt), std::optional<T>);
  assert(static_cast<bool>(opt));
  assert(*opt == arg);
}

constexpr bool test() {
  //  optional(T)
  test_deduct<int>(5);
  test_deduct<A>(A{});

  {
    //  optional(const T&);
    const int& source = 5;
    test_deduct<int>(source);
  }

  {
    //  optional(T*);
    const int* source = nullptr;
    test_deduct<const int*>(source);
  }

  {
    //  optional(T[]);
    int source[] = {1, 2, 3};
    std::optional opt(source);

    ASSERT_SAME_TYPE(decltype(opt), std::optional<int*>);
    assert(static_cast<bool>(opt));
    assert((*opt)[0] == 1);
  }

  //  Test the implicit deduction guides
  {
    //  optional(optional);
    std::optional<char> source('A');
    std::optional opt(source);

    ASSERT_SAME_TYPE(decltype(opt), std::optional<char>);
    assert(static_cast<bool>(opt) == static_cast<bool>(source));
    assert(*opt == *source);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
