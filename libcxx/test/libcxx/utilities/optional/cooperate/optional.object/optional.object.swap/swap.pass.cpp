//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// void swap(optional&)
//     noexcept(is_nothrow_move_constructible<T>::value &&
//              is_nothrow_swappable<T>::value)

#include <optional>
#include <type_traits>
#include <functional>
#include <cassert>

#include "test_macros.h"

using std::optional;

template <typename T>
struct equal_to {
  constexpr bool operator()(T& l, T& r) const noexcept { return l == r; }
};

template <typename T>
struct equal_to<std::reference_wrapper<T>> {
  constexpr bool operator()(std::reference_wrapper<T>& l, std::reference_wrapper<T>& r) const noexcept {
    return std::addressof(l.get()) == std::addressof(r.get());
  }
};

template <class T>
TEST_CONSTEXPR_CXX20 bool check_swap(T left, T right) {
  constexpr equal_to<T> is_equal;
  {
    optional<T> opt1;
    optional<T> opt2;
    static_assert(
        noexcept(opt1.swap(opt2)) == (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_swappable_v<T>));
    assert(static_cast<bool>(opt1) == false);
    assert(static_cast<bool>(opt2) == false);
    opt1.swap(opt2);
    assert(static_cast<bool>(opt1) == false);
    assert(static_cast<bool>(opt2) == false);
  }
  {
    optional<T> opt1(left);
    optional<T> opt2;
    static_assert(
        noexcept(opt1.swap(opt2)) == (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_swappable_v<T>));
    assert(static_cast<bool>(opt1) == true);
    assert(is_equal(*opt1, left));
    assert(static_cast<bool>(opt2) == false);
    opt1.swap(opt2);
    assert(static_cast<bool>(opt1) == false);
    assert(static_cast<bool>(opt2) == true);
    assert(is_equal(*opt2, left));
  }
  {
    optional<T> opt1;
    optional<T> opt2(right);
    static_assert(
        noexcept(opt1.swap(opt2)) == (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_swappable_v<T>));
    assert(static_cast<bool>(opt1) == false);
    assert(static_cast<bool>(opt2) == true);
    assert(is_equal(*opt2, right));
    opt1.swap(opt2);
    assert(static_cast<bool>(opt1) == true);
    assert(is_equal(*opt1, right));
    assert(static_cast<bool>(opt2) == false);
  }
  {
    optional<T> opt1(left);
    optional<T> opt2(right);
    static_assert(
        noexcept(opt1.swap(opt2)) == (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_swappable_v<T>));
    assert(static_cast<bool>(opt1) == true);
    assert(is_equal(*opt1, left));
    assert(static_cast<bool>(opt2) == true);
    assert(is_equal(*opt2, right));
    opt1.swap(opt2);
    assert(static_cast<bool>(opt1) == true);
    assert(is_equal(*opt1, right));
    assert(static_cast<bool>(opt2) == true);
    assert(is_equal(*opt2, left));
  }
  return true;
}

int f() noexcept { return 0; }
int g() noexcept { return 0; }

int main(int, char**) {
  int i, j;
  check_swap<std::reference_wrapper<int>>(i, j);
  check_swap<std::reference_wrapper<const int>>(i, j);
  check_swap<std::reference_wrapper<int()>>(f, g);
  check_swap<std::reference_wrapper<int() noexcept>>(f, g);
#if TEST_STD_VER > 17
  static int ii, jj;
  static_assert(check_swap<std::reference_wrapper<const int>>(ii, jj));
  static_assert(check_swap<std::reference_wrapper<int()>>(f, g));
#endif
}
