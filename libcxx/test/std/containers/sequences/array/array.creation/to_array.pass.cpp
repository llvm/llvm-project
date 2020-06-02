//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// template <typename T, size_t Size>
// constexpr auto to_array(T (&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

// template <typename T, size_t Size>
// constexpr auto to_array(T (&&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

#include <array>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests()
{
  //  Test deduced type.
  {
    auto arr = std::to_array({1, 2, 3});
    ASSERT_SAME_TYPE(decltype(arr), std::array<int, 3>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    const long l1 = 42;
    auto arr = std::to_array({1L, 4L, 9L, l1});
    ASSERT_SAME_TYPE(decltype(arr)::value_type, long);
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
  }

  {
    auto arr = std::to_array("meow");
    ASSERT_SAME_TYPE(decltype(arr), std::array<char, 5>);
    assert(arr[0] == 'm');
    assert(arr[1] == 'e');
    assert(arr[2] == 'o');
    assert(arr[3] == 'w');
    assert(arr[4] == '\0');
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr = std::to_array(source);
    ASSERT_SAME_TYPE(decltype(arr), std::array<double, 3>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr = std::to_array(std::move(source));
    ASSERT_SAME_TYPE(decltype(arr), std::array<double, 3>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

  {
    MoveOnly source[] = {MoveOnly{0}, MoveOnly{1}, MoveOnly{2}};

    auto arr = std::to_array(std::move(source));
    ASSERT_SAME_TYPE(decltype(arr), std::array<MoveOnly, 3>);
    for (int i = 0; i < 3; ++i)
      assert(arr[i].get() == i && source[i].get() == 0);
  }

  // Test C99 compound literal.
  {
    auto arr = std::to_array((int[]){3, 4});
    ASSERT_SAME_TYPE(decltype(arr), std::array<int, 2>);
    assert(arr[0] == 3);
    assert(arr[1] == 4);
  }

  //  Test explicit type.
  {
    auto arr = std::to_array<long>({1, 2, 3});
    ASSERT_SAME_TYPE(decltype(arr), std::array<long, 3>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    struct A {
      int a;
      double b;
    };

    auto arr = std::to_array<A>({{3, .1}});
    ASSERT_SAME_TYPE(decltype(arr), std::array<A, 1>);
    assert(arr[0].a == 3);
    assert(arr[0].b == .1);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
