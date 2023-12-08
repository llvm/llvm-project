//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "test_macros.h"

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__type_traits/is_valid_expansion.h>
#include <type_traits>
#include <cassert>

struct Bomb;
template <int N, class T = Bomb >
struct BOOM {
  using Explode = typename T::BOOMBOOM;
};

using True = std::true_type;
using False = std::false_type;

void test_if() {
  ASSERT_SAME_TYPE(std::_If<true, int, long>, int);
  ASSERT_SAME_TYPE(std::_If<false, int, long>, long);
}

void test_and() {
  static_assert(std::_And<True>::value, "");
  static_assert(!std::_And<False>::value, "");
  static_assert(std::_And<True, True>::value, "");
  static_assert(!std::_And<False, BOOM<1> >::value, "");
  static_assert(!std::_And<True, True, True, False, BOOM<2> >::value, "");
}

void test_or() {
  static_assert(std::_Or<True>::value, "");
  static_assert(!std::_Or<False>::value, "");
  static_assert(std::_Or<False, True>::value, "");
  static_assert(std::_Or<True, std::_Not<BOOM<3> > >::value, "");
  static_assert(!std::_Or<False, False>::value, "");
  static_assert(std::_Or<True, BOOM<1> >::value, "");
  static_assert(std::_Or<False, False, False, False, True, BOOM<2> >::value, "");
}

void test_combined() {
  static_assert(std::_And<True, std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(std::_And<True, std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(std::_Not<std::_And<True, False, BOOM<5> > >::value, "");
}

struct MemberTest {
  static int foo;
  using type = long;

  void func(int);
};
struct Empty {};
struct MemberTest2 {
  using foo = int;
};
template <class T>
using HasFooData = decltype(T::foo);
template <class T>
using HasFooType = typename T::foo;

template <class T, class U>
using FuncCallable = decltype(std::declval<T>().func(std::declval<U>()));
template <class T>
using BadCheck = typename T::DOES_NOT_EXIST;

void test_is_valid_trait() {
  static_assert(std::_IsValidExpansion<HasFooData, MemberTest>::value, "");
  static_assert(!std::_IsValidExpansion<HasFooType, MemberTest>::value, "");
  static_assert(!std::_IsValidExpansion<HasFooData, MemberTest2>::value, "");
  static_assert(std::_IsValidExpansion<HasFooType, MemberTest2>::value, "");
  static_assert(std::_IsValidExpansion<FuncCallable, MemberTest, int>::value, "");
  static_assert(!std::_IsValidExpansion<FuncCallable, MemberTest, void*>::value, "");
}

int main(int, char**) {
  return 0;
}
