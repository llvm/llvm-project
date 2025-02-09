//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<ValueType T, size_t N>
//   requires Swappable<T>
// void swap(T (&a)[N], T (&b)[N]); // constexpr since C++20

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"

#if TEST_STD_VER >= 11
struct CopyOnly {
  TEST_CONSTEXPR_CXX20 CopyOnly() {}
  TEST_CONSTEXPR_CXX20 CopyOnly(CopyOnly const&) noexcept {}
  TEST_CONSTEXPR_CXX20 CopyOnly& operator=(CopyOnly const&) { return *this; }
};

struct NoexceptMoveOnly {
  TEST_CONSTEXPR_CXX20 NoexceptMoveOnly() {}
  TEST_CONSTEXPR_CXX20 NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
  TEST_CONSTEXPR_CXX20 NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept { return *this; }
};

struct NotMoveConstructible {
  TEST_CONSTEXPR_CXX20 NotMoveConstructible() {}
  TEST_CONSTEXPR_CXX20 NotMoveConstructible& operator=(NotMoveConstructible&&) { return *this; }
  NotMoveConstructible(NotMoveConstructible&&) = delete;
};

template <class Tp>
auto can_swap_test(int) -> decltype(std::swap(std::declval<Tp>(), std::declval<Tp>()));

template <class Tp>
auto can_swap_test(...) -> std::false_type;

template <class Tp>
constexpr bool can_swap() {
  return std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}
#endif

#if TEST_STD_VER >= 11
// This test is constexpr only since C++23 because constexpr std::unique_ptr is only available since C++23
TEST_CONSTEXPR_CXX23 bool test_unique_ptr() {
  std::unique_ptr<int> i[3];
  for (int k = 0; k < 3; ++k)
    i[k].reset(new int(k + 1));
  std::unique_ptr<int> j[3];
  for (int k = 0; k < 3; ++k)
    j[k].reset(new int(k + 4));
  std::swap(i, j);
  assert(*i[0] == 4);
  assert(*i[1] == 5);
  assert(*i[2] == 6);
  assert(*j[0] == 1);
  assert(*j[1] == 2);
  assert(*j[2] == 3);
  return true;
}
#endif

TEST_CONSTEXPR_CXX20 bool test() {
  {
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    std::swap(i, j);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
  }
  {
    int a[2][2]   = {{0, 1}, {2, 3}};
    decltype(a) b = {{9, 8}, {7, 6}};

    std::swap(a, b);

    assert(a[0][0] == 9);
    assert(a[0][1] == 8);
    assert(a[1][0] == 7);
    assert(a[1][1] == 6);

    assert(b[0][0] == 0);
    assert(b[0][1] == 1);
    assert(b[1][0] == 2);
    assert(b[1][1] == 3);
  }

  {
    int a[3][3]   = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    decltype(a) b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    std::swap(a, b);

    assert(a[0][0] == 9);
    assert(a[0][1] == 8);
    assert(a[0][2] == 7);
    assert(a[1][0] == 6);
    assert(a[1][1] == 5);
    assert(a[1][2] == 4);
    assert(a[2][0] == 3);
    assert(a[2][1] == 2);
    assert(a[2][2] == 1);

    assert(b[0][0] == 0);
    assert(b[0][1] == 1);
    assert(b[0][2] == 2);
    assert(b[1][0] == 3);
    assert(b[1][1] == 4);
    assert(b[1][2] == 5);
    assert(b[2][0] == 6);
    assert(b[2][1] == 7);
    assert(b[2][2] == 8);
  }
#if TEST_STD_VER >= 11
  {
    using CA = CopyOnly[42];
    using MA = NoexceptMoveOnly[42];
    using NA = NotMoveConstructible[42];
    static_assert(can_swap<CA&>(), "");
    static_assert(can_swap<MA&>(), "");
    static_assert(!can_swap<NA&>(), "");

    CA ca;
    MA ma;
    static_assert(!noexcept(std::swap(ca, ca)), "");
    static_assert(noexcept(std::swap(ma, ma)), "");
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 11
  test_unique_ptr();
#endif
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
#if TEST_STD_VER >= 23
  static_assert(test_unique_ptr());
#endif

  return 0;
}
