//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <bitset>

// class bitset<N>::reference
// friend void swap(reference x, reference y);
// friend void swap(reference x, bool&);
// friend void swap(bool& x, reference y);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <cstddef>
#include <bitset>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class T, class U, class = void>
struct can_qualified_std_swap_with : std::false_type {};
template <class T, class U>
struct can_qualified_std_swap_with<T, U, decltype((void)std::swap(std::declval<T>(), std::declval<U>()))>
    : std::true_type {};

template <class T>
struct can_qualified_std_swap : can_qualified_std_swap_with<T&, T&>::type {};

namespace adl_only {
void swap();

template <class T, class U, class = void>
struct can_swap_with : std::false_type {};
template <class T, class U>
struct can_swap_with<T, U, decltype((void)swap(std::declval<T>(), std::declval<U>()))> : std::true_type {};

template <class T>
struct can_swap : can_swap_with<T&, T&>::type {};
} // namespace adl_only

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test() {
  typedef typename std::bitset<N>::reference BRef;

  // Test that only homogeneous bitset<N>::reference swap is supported.
  typedef std::bitset<1>::reference BRef1;
  static_assert(can_qualified_std_swap_with<BRef1, BRef>::value == std::is_same<BRef1, BRef>::value, "");
  static_assert(adl_only::can_swap_with<BRef1, BRef>::value == std::is_same<BRef1, BRef>::value, "");

  static_assert(can_qualified_std_swap<BRef>::value, "");
  static_assert(!can_qualified_std_swap_with<BRef, BRef>::value, "");
  static_assert(!can_qualified_std_swap_with<BRef, bool&>::value, "");
  static_assert(!can_qualified_std_swap_with<bool&, BRef>::value, "");
  static_assert(adl_only::can_swap<BRef>::value, "");
  static_assert(adl_only::can_swap_with<BRef, BRef>::value, "");
  static_assert(adl_only::can_swap_with<BRef, bool&>::value, "");
  static_assert(adl_only::can_swap_with<bool&, BRef>::value, "");

  using std::swap;

  std::bitset<N> v;
  BRef r1 = v[0];
  BRef r2 = v[1];
  r1      = true;
  r2      = false;

  swap(r1, r2);
  assert(v[0] == false);
  assert(v[1] == true);

  bool b1 = true;
  swap(r1, b1);
  assert(v[0] == true);
  assert(b1 == false);

  swap(b1, r1);
  assert(v[0] == false);
  assert(b1 == true);
}

TEST_CONSTEXPR_CXX23 bool test() {
  test<3>();
  test<4>();
  test<6>();
  test<8>();
  test<12>();
  test<16>();
  test<24>();
  test<32>();
  test<48>();
  test<64>();
  test<96>();
  test<128>();
  test<192>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
  return 0;
}
