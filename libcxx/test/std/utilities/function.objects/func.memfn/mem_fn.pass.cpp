//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class R, class T> constexpr unspecified mem_fn(R T::*) noexcept;       // constexpr in C++20

#include <functional>
#include <cassert>
#include <utility>
#include <type_traits>

#include "test_macros.h"

struct A {
  double data_;

  TEST_CONSTEXPR_CXX14 char test0() { return 'a'; }
  TEST_CONSTEXPR_CXX14 char test1(int) { return 'b'; }
  TEST_CONSTEXPR_CXX14 char test2(int, double) { return 'c'; }

  TEST_CONSTEXPR_CXX14 char test0_nothrow() TEST_NOEXCEPT { return 'd'; }
  TEST_CONSTEXPR_CXX14 char test1_nothrow(int) TEST_NOEXCEPT { return 'e'; }
  TEST_CONSTEXPR_CXX14 char test2_nothrow(int, double) TEST_NOEXCEPT { return 'f'; }

  TEST_CONSTEXPR char test_c0() const { return 'a'; }
  TEST_CONSTEXPR char test_c1(int) const { return 'b'; }
  TEST_CONSTEXPR char test_c2(int, double) const { return 'c'; }

  TEST_CONSTEXPR char test_c0_nothrow() const TEST_NOEXCEPT { return 'd'; }
  TEST_CONSTEXPR char test_c1_nothrow(int) const TEST_NOEXCEPT { return 'e'; }
  TEST_CONSTEXPR char test_c2_nothrow(int, double) const TEST_NOEXCEPT { return 'f'; }

  char test_v0() volatile { return 'a'; }
  char test_v1(int) volatile { return 'b'; }
  char test_v2(int, double) volatile { return 'c'; }

  char test_v0_nothrow() volatile TEST_NOEXCEPT { return 'd'; }
  char test_v1_nothrow(int) volatile TEST_NOEXCEPT { return 'e'; }
  char test_v2_nothrow(int, double) volatile TEST_NOEXCEPT { return 'f'; }

  char test_cv0() const volatile { return 'a'; }
  char test_cv1(int) const volatile { return 'b'; }
  char test_cv2(int, double) const volatile { return 'c'; }

  char test_cv0_nothrow() const volatile TEST_NOEXCEPT { return 'd'; }
  char test_cv1_nothrow(int) const volatile TEST_NOEXCEPT { return 'e'; }
  char test_cv2_nothrow(int, double) const volatile TEST_NOEXCEPT { return 'f'; }
};

template <class F>
TEST_CONSTEXPR_CXX20 bool test_data(F f) {
  A a  = {0.0};
  f(a) = 5;
  assert(a.data_ == 5);
  A* ap = &a;
  f(ap) = 6;
  assert(a.data_ == 6);
  const A* cap = ap;
  assert(f(cap) == f(ap));
  const F& cf = f;
  assert(cf(ap) == f(ap));

#if TEST_STD_VER >= 11
  static_assert(noexcept(f(a)), "");
  static_assert(noexcept(f(ap)), "");
  static_assert(noexcept(f(cap)), "");
  static_assert(noexcept(cf(ap)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_fun0(F f) {
  A a = {};
  assert(f(a) == 'a');
  A* ap = &a;
  assert(f(ap) == 'a');
  const F& cf = f;
  assert(cf(ap) == 'a');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a)), "");
  static_assert(!noexcept(f(ap)), "");
  static_assert(!noexcept(cf(ap)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'b');
  A* ap = &a;
  assert(f(ap, 2) == 'b');
  const F& cf = f;
  assert(cf(ap, 2) == 'b');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0)), "");
  static_assert(!noexcept(f(ap, 1)), "");
  static_assert(!noexcept(cf(ap, 2)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'c');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'c');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'c');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0, 0.0)), "");
  static_assert(!noexcept(f(ap, 1, 2)), "");
  static_assert(!noexcept(cf(ap, 2, 3.5)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_noexcept_fun0(F f) {
  A a = {};
  assert(f(a) == 'd');
  A* ap = &a;
  assert(f(ap) == 'd');
  const F& cf = f;
  assert(cf(ap) == 'd');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a)), "");
  static_assert(noexcept(f(ap)), "");
  static_assert(noexcept(cf(ap)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_noexcept_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'e');
  A* ap = &a;
  assert(f(ap, 2) == 'e');
  const F& cf = f;
  assert(cf(ap, 2) == 'e');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0)), "");
  static_assert(noexcept(f(ap, 1)), "");
  static_assert(noexcept(cf(ap, 2)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_noexcept_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'f');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'f');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'f');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0, 0.0)), "");
  static_assert(noexcept(f(ap, 1, 2)), "");
  static_assert(noexcept(cf(ap, 2, 3.5)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_fun0(F f) {
  A a = {};
  assert(f(a) == 'a');
  A* ap = &a;
  assert(f(ap) == 'a');
  const A* cap = &a;
  assert(f(cap) == 'a');
  const F& cf = f;
  assert(cf(ap) == 'a');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a)), "");
  static_assert(!noexcept(f(ap)), "");
  static_assert(!noexcept(f(cap)), "");
  static_assert(!noexcept(cf(ap)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'b');
  A* ap = &a;
  assert(f(ap, 2) == 'b');
  const A* cap = &a;
  assert(f(cap, 2) == 'b');
  const F& cf = f;
  assert(cf(ap, 2) == 'b');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0)), "");
  static_assert(!noexcept(f(ap, 1)), "");
  static_assert(!noexcept(f(cap, 2)), "");
  static_assert(!noexcept(cf(ap, 3)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'c');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'c');
  const A* cap = &a;
  assert(f(cap, 2, 3.5) == 'c');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'c');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0, 0.0)), "");
  static_assert(!noexcept(f(ap, 1, 2)), "");
  static_assert(!noexcept(f(cap, 2, 3.5)), "");
  static_assert(!noexcept(cf(ap, 3, 17.29)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_noexcept_fun0(F f) {
  A a = {};
  assert(f(a) == 'd');
  A* ap = &a;
  assert(f(ap) == 'd');
  const A* cap = &a;
  assert(f(cap) == 'd');
  const F& cf = f;
  assert(cf(ap) == 'd');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a)), "");
  static_assert(noexcept(f(ap)), "");
  static_assert(noexcept(f(cap)), "");
  static_assert(noexcept(cf(ap)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_noexcept_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'e');
  A* ap = &a;
  assert(f(ap, 2) == 'e');
  const A* cap = &a;
  assert(f(cap, 2) == 'e');
  const F& cf = f;
  assert(cf(ap, 2) == 'e');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0)), "");
  static_assert(noexcept(f(ap, 1)), "");
  static_assert(noexcept(f(cap, 2)), "");
  static_assert(noexcept(cf(ap, 3)), "");
#endif

  return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool test_const_noexcept_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'f');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'f');
  const A* cap = &a;
  assert(f(cap, 2, 3.5) == 'f');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'f');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0, 0.0)), "");
  static_assert(noexcept(f(ap, 1, 2)), "");
  static_assert(noexcept(f(cap, 2, 3.5)), "");
  static_assert(noexcept(cf(ap, 3, 17.29)), "");
#endif

  return true;
}

template <class F>
void test_volatile_fun0(F f) {
  A a = {};
  assert(f(a) == 'a');
  A* ap = &a;
  assert(f(ap) == 'a');
  volatile A* cap = &a;
  assert(f(cap) == 'a');
  const F& cf = f;
  assert(cf(ap) == 'a');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a)), "");
  static_assert(!noexcept(f(ap)), "");
  static_assert(!noexcept(f(cap)), "");
  static_assert(!noexcept(cf(ap)), "");
#endif
}

template <class F>
void test_volatile_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'b');
  A* ap = &a;
  assert(f(ap, 2) == 'b');
  volatile A* cap = &a;
  assert(f(cap, 2) == 'b');
  const F& cf = f;
  assert(cf(ap, 2) == 'b');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0)), "");
  static_assert(!noexcept(f(ap, 1)), "");
  static_assert(!noexcept(f(cap, 2)), "");
  static_assert(!noexcept(cf(ap, 3)), "");
#endif
}

template <class F>
void test_volatile_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'c');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'c');
  volatile A* cap = &a;
  assert(f(cap, 2, 3.5) == 'c');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'c');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0, 0.0)), "");
  static_assert(!noexcept(f(ap, 1, 2)), "");
  static_assert(!noexcept(f(cap, 2, 3.5)), "");
  static_assert(!noexcept(cf(ap, 3, 17.29)), "");
#endif
}

template <class F>
void test_volatile_noexcept_fun0(F f) {
  A a = {};
  assert(f(a) == 'd');
  A* ap = &a;
  assert(f(ap) == 'd');
  volatile A* cap = &a;
  assert(f(cap) == 'd');
  const F& cf = f;
  assert(cf(ap) == 'd');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a)), "");
  static_assert(noexcept(f(ap)), "");
  static_assert(noexcept(f(cap)), "");
  static_assert(noexcept(cf(ap)), "");
#endif
}

template <class F>
void test_volatile_noexcept_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'e');
  A* ap = &a;
  assert(f(ap, 2) == 'e');
  volatile A* cap = &a;
  assert(f(cap, 2) == 'e');
  const F& cf = f;
  assert(cf(ap, 2) == 'e');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0)), "");
  static_assert(noexcept(f(ap, 1)), "");
  static_assert(noexcept(f(cap, 2)), "");
  static_assert(noexcept(cf(ap, 3)), "");
#endif
}

template <class F>
void test_volatile_noexcept_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'f');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'f');
  volatile A* cap = &a;
  assert(f(cap, 2, 3.5) == 'f');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'f');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0, 0.0)), "");
  static_assert(noexcept(f(ap, 1, 2)), "");
  static_assert(noexcept(f(cap, 2, 3.5)), "");
  static_assert(noexcept(cf(ap, 3, 17.29)), "");
#endif
}

template <class F>
void test_const_volatile_fun0(F f) {
  A a = {};
  assert(f(a) == 'a');
  A* ap = &a;
  assert(f(ap) == 'a');
  const volatile A* cap = &a;
  assert(f(cap) == 'a');
  const F& cf = f;
  assert(cf(ap) == 'a');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a)), "");
  static_assert(!noexcept(f(ap)), "");
  static_assert(!noexcept(f(cap)), "");
  static_assert(!noexcept(cf(ap)), "");
#endif
}

template <class F>
void test_const_volatile_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'b');
  A* ap = &a;
  assert(f(ap, 2) == 'b');
  const volatile A* cap = &a;
  assert(f(cap, 2) == 'b');
  const F& cf = f;
  assert(cf(ap, 2) == 'b');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0)), "");
  static_assert(!noexcept(f(ap, 1)), "");
  static_assert(!noexcept(f(cap, 2)), "");
  static_assert(!noexcept(cf(ap, 3)), "");
#endif
}

template <class F>
void test_const_volatile_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'c');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'c');
  const volatile A* cap = &a;
  assert(f(cap, 2, 3.5) == 'c');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'c');

#if TEST_STD_VER >= 17
  static_assert(!noexcept(f(a, 0, 0.0)), "");
  static_assert(!noexcept(f(ap, 1, 2)), "");
  static_assert(!noexcept(f(cap, 2, 3.5)), "");
  static_assert(!noexcept(cf(ap, 3, 17.29)), "");
#endif
}

template <class F>
void test_const_volatile_noexcept_fun0(F f) {
  A a = {};
  assert(f(a) == 'd');
  A* ap = &a;
  assert(f(ap) == 'd');
  const volatile A* cap = &a;
  assert(f(cap) == 'd');
  const F& cf = f;
  assert(cf(ap) == 'd');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a)), "");
  static_assert(noexcept(f(ap)), "");
  static_assert(noexcept(f(cap)), "");
  static_assert(noexcept(cf(ap)), "");
#endif
}

template <class F>
void test_const_volatile_noexcept_fun1(F f) {
  A a = {};
  assert(f(a, 1) == 'e');
  A* ap = &a;
  assert(f(ap, 2) == 'e');
  const volatile A* cap = &a;
  assert(f(cap, 2) == 'e');
  const F& cf = f;
  assert(cf(ap, 2) == 'e');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0)), "");
  static_assert(noexcept(f(ap, 1)), "");
  static_assert(noexcept(f(cap, 2)), "");
  static_assert(noexcept(cf(ap, 3)), "");
#endif
}

template <class F>
void test_const_volatile_noexcept_fun2(F f) {
  A a = {};
  assert(f(a, 1, 2) == 'f');
  A* ap = &a;
  assert(f(ap, 2, 3.5) == 'f');
  const volatile A* cap = &a;
  assert(f(cap, 2, 3.5) == 'f');
  const F& cf = f;
  assert(cf(ap, 2, 3.5) == 'f');

#if TEST_STD_VER >= 17
  static_assert(noexcept(f(a, 0, 0.0)), "");
  static_assert(noexcept(f(ap, 1, 2)), "");
  static_assert(noexcept(f(cap, 2, 3.5)), "");
  static_assert(noexcept(cf(ap, 3, 17.29)), "");
#endif
}

#if TEST_STD_VER >= 11
template <class V, class Func, class... Args>
struct is_callable_impl : std::false_type {};

template <class Func, class... Args>
struct is_callable_impl<decltype((void)std::declval<Func>()(std::declval<Args>()...)), Func, Args...> : std::true_type {
};

template <class Func, class... Args>
struct is_callable : is_callable_impl<void, Func, Args...>::type {};

template <class F>
void test_sfinae_data(F) {
  static_assert(is_callable<F, A>::value, "");
  static_assert(is_callable<F, const A>::value, "");
  static_assert(is_callable<F, A&>::value, "");
  static_assert(is_callable<F, const A&>::value, "");
  static_assert(is_callable<F, A*>::value, "");
  static_assert(is_callable<F, const A*>::value, "");

  static_assert(!is_callable<F, A, char>::value, "");
  static_assert(!is_callable<F, const A, char>::value, "");
  static_assert(!is_callable<F, A&, char>::value, "");
  static_assert(!is_callable<F, const A&, char>::value, "");
  static_assert(!is_callable<F, A*, char>::value, "");
  static_assert(!is_callable<F, const A*, char>::value, "");
}

template <class F>
void test_sfinae_fun0(F) {
  static_assert(is_callable<F, A>::value, "");
  static_assert(is_callable<F, A&>::value, "");
  static_assert(is_callable<F, A*>::value, "");

  static_assert(!is_callable<F, const A>::value, "");
  static_assert(!is_callable<F, const A&>::value, "");
  static_assert(!is_callable<F, const A*>::value, "");

  static_assert(!is_callable<F, volatile A>::value, "");
  static_assert(!is_callable<F, volatile A&>::value, "");
  static_assert(!is_callable<F, volatile A*>::value, "");

  static_assert(!is_callable<F, const volatile A>::value, "");
  static_assert(!is_callable<F, const volatile A&>::value, "");
  static_assert(!is_callable<F, const volatile A*>::value, "");

  static_assert(!is_callable<F, A, int>::value, "");
  static_assert(!is_callable<F, A&, int>::value, "");
  static_assert(!is_callable<F, A*, int>::value, "");
}

template <class F>
void test_sfinae_fun1(F) {
  static_assert(is_callable<F, A, int>::value, "");
  static_assert(is_callable<F, A&, int>::value, "");
  static_assert(is_callable<F, A*, int>::value, "");

  static_assert(!is_callable<F, A>::value, "");
  static_assert(!is_callable<F, A&>::value, "");
  static_assert(!is_callable<F, A*>::value, "");
}

template <class F>
void test_sfinae_const_fun0(F) {
  static_assert(is_callable<F, A>::value, "");
  static_assert(is_callable<F, A&>::value, "");
  static_assert(is_callable<F, A*>::value, "");

  static_assert(is_callable<F, const A>::value, "");
  static_assert(is_callable<F, const A&>::value, "");
  static_assert(is_callable<F, const A*>::value, "");

  static_assert(!is_callable<F, volatile A>::value, "");
  static_assert(!is_callable<F, volatile A&>::value, "");
  static_assert(!is_callable<F, volatile A*>::value, "");

  static_assert(!is_callable<F, const volatile A>::value, "");
  static_assert(!is_callable<F, const volatile A&>::value, "");
  static_assert(!is_callable<F, const volatile A*>::value, "");
}

template <class F>
void test_sfinae_volatile_fun0(F) {
  static_assert(is_callable<F, A>::value, "");
  static_assert(is_callable<F, A&>::value, "");
  static_assert(is_callable<F, A*>::value, "");

  static_assert(!is_callable<F, const A>::value, "");
  static_assert(!is_callable<F, const A&>::value, "");
  static_assert(!is_callable<F, const A*>::value, "");

  static_assert(is_callable<F, volatile A>::value, "");
  static_assert(is_callable<F, volatile A&>::value, "");
  static_assert(is_callable<F, volatile A*>::value, "");

  static_assert(!is_callable<F, const volatile A>::value, "");
  static_assert(!is_callable<F, const volatile A&>::value, "");
  static_assert(!is_callable<F, const volatile A*>::value, "");
}

template <class F>
void test_sfinae_const_volatile_fun0(F) {
  static_assert(is_callable<F, A>::value, "");
  static_assert(is_callable<F, A&>::value, "");
  static_assert(is_callable<F, A*>::value, "");

  static_assert(is_callable<F, const A>::value, "");
  static_assert(is_callable<F, const A&>::value, "");
  static_assert(is_callable<F, const A*>::value, "");

  static_assert(is_callable<F, volatile A>::value, "");
  static_assert(is_callable<F, volatile A&>::value, "");
  static_assert(is_callable<F, volatile A*>::value, "");

  static_assert(is_callable<F, const volatile A>::value, "");
  static_assert(is_callable<F, const volatile A&>::value, "");
  static_assert(is_callable<F, const volatile A*>::value, "");
}
#endif

int main(int, char**) {
  test_data(std::mem_fn(&A::data_));

  test_fun0(std::mem_fn(&A::test0));
  test_fun1(std::mem_fn(&A::test1));
  test_fun2(std::mem_fn(&A::test2));

  test_noexcept_fun0(std::mem_fn(&A::test0_nothrow));
  test_noexcept_fun1(std::mem_fn(&A::test1_nothrow));
  test_noexcept_fun2(std::mem_fn(&A::test2_nothrow));

  test_const_fun0(std::mem_fn(&A::test_c0));
  test_const_fun1(std::mem_fn(&A::test_c1));
  test_const_fun2(std::mem_fn(&A::test_c2));

  test_const_noexcept_fun0(std::mem_fn(&A::test_c0_nothrow));
  test_const_noexcept_fun1(std::mem_fn(&A::test_c1_nothrow));
  test_const_noexcept_fun2(std::mem_fn(&A::test_c2_nothrow));

  test_volatile_fun0(std::mem_fn(&A::test_v0));
  test_volatile_fun1(std::mem_fn(&A::test_v1));
  test_volatile_fun2(std::mem_fn(&A::test_v2));

  test_volatile_noexcept_fun0(std::mem_fn(&A::test_v0_nothrow));
  test_volatile_noexcept_fun1(std::mem_fn(&A::test_v1_nothrow));
  test_volatile_noexcept_fun2(std::mem_fn(&A::test_v2_nothrow));

  test_const_volatile_fun0(std::mem_fn(&A::test_cv0));
  test_const_volatile_fun1(std::mem_fn(&A::test_cv1));
  test_const_volatile_fun2(std::mem_fn(&A::test_cv2));

  test_const_volatile_noexcept_fun0(std::mem_fn(&A::test_cv0_nothrow));
  test_const_volatile_noexcept_fun1(std::mem_fn(&A::test_cv1_nothrow));
  test_const_volatile_noexcept_fun2(std::mem_fn(&A::test_cv2_nothrow));

#if TEST_STD_VER >= 11
  // LWG2489
  static_assert((noexcept(std::mem_fn(&A::data_))), "");
  static_assert((noexcept(std::mem_fn(&A::test0))), "");
  static_assert((noexcept(std::mem_fn(&A::test0_nothrow))), "");

  test_sfinae_data(std::mem_fn(&A::data_));

  test_sfinae_fun0(std::mem_fn(&A::test0));
  test_sfinae_fun0(std::mem_fn(&A::test0_nothrow));

  test_sfinae_const_fun0(std::mem_fn(&A::test_c0));
  test_sfinae_const_fun0(std::mem_fn(&A::test_c0_nothrow));

  test_sfinae_volatile_fun0(std::mem_fn(&A::test_v0));
  test_sfinae_volatile_fun0(std::mem_fn(&A::test_v0_nothrow));

  test_sfinae_const_volatile_fun0(std::mem_fn(&A::test_cv0));
  test_sfinae_const_volatile_fun0(std::mem_fn(&A::test_cv0_nothrow));

  test_sfinae_fun1(std::mem_fn(&A::test1));
  test_sfinae_fun1(std::mem_fn(&A::test1_nothrow));
#endif

#if TEST_STD_VER >= 20
  static_assert(test_data(std::mem_fn(&A::data_)));

  static_assert(test_fun0(std::mem_fn(&A::test0)));
  static_assert(test_fun1(std::mem_fn(&A::test1)));
  static_assert(test_fun2(std::mem_fn(&A::test2)));

  static_assert(test_const_fun0(std::mem_fn(&A::test_c0)));
  static_assert(test_const_fun1(std::mem_fn(&A::test_c1)));
  static_assert(test_const_fun2(std::mem_fn(&A::test_c2)));

  static_assert(test_noexcept_fun0(std::mem_fn(&A::test0_nothrow)));
  static_assert(test_noexcept_fun1(std::mem_fn(&A::test1_nothrow)));
  static_assert(test_noexcept_fun2(std::mem_fn(&A::test2_nothrow)));

  static_assert(test_const_noexcept_fun0(std::mem_fn(&A::test_c0_nothrow)));
  static_assert(test_const_noexcept_fun1(std::mem_fn(&A::test_c1_nothrow)));
  static_assert(test_const_noexcept_fun2(std::mem_fn(&A::test_c2_nothrow)));
#endif

  return 0;
}
