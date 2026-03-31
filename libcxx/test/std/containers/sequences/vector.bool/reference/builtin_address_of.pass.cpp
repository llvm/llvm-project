//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// template<class Allocator>
// class vector<bool, Allocator>::reference;

// Verify that vector<bool, Allocator>::reference has no overloaded operator&.

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "min_allocator.h"

namespace test_overloads {

struct dummy {};
void operator&(dummy);

template <class, class = void>
struct has_nonmember_operator_address_of : std::false_type {};
template <class T>
struct has_nonmember_operator_address_of<T, decltype((void)operator&(std::declval<T>()))> : std::true_type {};

template <class, class = void>
struct has_member_operator_address_of : std::false_type {};
template <class T>
struct has_member_operator_address_of<T, decltype((void)std::declval<T>().operator&())> : std::true_type {};

} // namespace test_overloads

template <class, class = void>
struct can_take_address : std::false_type {};
template <class T>
struct can_take_address<T, decltype((void)&std::declval<T>())> : std::true_type {};

template <class, class = void>
struct has_builtin_operator_address_of : std::false_type {};
template <class T>
struct has_builtin_operator_address_of<T, decltype((void)&std::declval<T>())>
    : std::integral_constant<bool,
                             !test_overloads::has_nonmember_operator_address_of<T>::value &&
                                 !test_overloads::has_member_operator_address_of<T>::value> {};

template <class Ref>
TEST_CONSTEXPR_CXX20 void test_proxy_references(Ref& r1, Ref& r2) {
  static_assert(can_take_address<Ref&>::value, "");
  static_assert(can_take_address<const Ref&>::value, "");
  static_assert(!can_take_address<Ref>::value, "");
  static_assert(!can_take_address<const Ref>::value, "");

  static_assert(has_builtin_operator_address_of<Ref&>::value, "");
  static_assert(has_builtin_operator_address_of<const Ref&>::value, "");
  static_assert(!has_builtin_operator_address_of<Ref>::value, "");
  static_assert(!has_builtin_operator_address_of<const Ref>::value, "");

  static_assert(std::is_same<decltype(&r1), Ref*>::value, "");

  assert(std::addressof(r1) == &r1);
  assert(std::addressof(r2) == &r2);
  assert((std::addressof(r1) == std::addressof(r2)) == (&r1 == &r2));
}

template <class VB>
TEST_CONSTEXPR_CXX20 void test() {
  VB vec(1);

  typename VB::reference r1 = vec[0];
  typename VB::reference r2 = vec[0];
  test_proxy_references(r1, r2);
  assert(&r1 != &r2);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test<std::vector<bool> >();
#if TEST_STD_VER >= 11
  test<std::vector<bool, min_allocator<bool>>>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
