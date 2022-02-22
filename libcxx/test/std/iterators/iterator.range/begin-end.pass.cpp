//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());
// template <class C> constexpr auto cbegin(const C& c) -> decltype(std::begin(c)); // C++14
// template <class C> constexpr auto cend(const C& c) -> decltype(std::end(c));     // C++14
// template <class C> constexpr auto end  (C& c) -> decltype(c.end());
// template <class C> constexpr auto end  (const C& c) -> decltype(c.end());
// template <class E> constexpr reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> constexpr reverse_iterator<const E*> rend  (initializer_list<E> il);
//
// template <class C> auto constexpr rbegin(C& c) -> decltype(c.rbegin());                 // C++14
// template <class C> auto constexpr rbegin(const C& c) -> decltype(c.rbegin());           // C++14
// template <class C> auto constexpr rend(C& c) -> decltype(c.rend());                     // C++14
// template <class C> constexpr auto rend(const C& c) -> decltype(c.rend());               // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rbegin(T (&array)[N]);      // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rend(T (&array)[N]);        // C++14
// template <class C> constexpr auto crbegin(const C& c) -> decltype(std::rbegin(c));      // C++14
// template <class C> constexpr auto crend(const C& c) -> decltype(std::rend(c));          // C++14
//
//  All of these are constexpr in C++17

#include <array>
#include <cassert>
#include <initializer_list>
#include <iterator>
#include <list>
#include <vector>

#include "test_macros.h"

// Throughout this test, we consistently compare against c.begin() instead of c.cbegin()
// because some STL types (e.g. initializer_list) don't syntactically support il.cbegin().
// Note that std::cbegin(x) effectively calls std::as_const(x).begin(), not x.cbegin();
// see the ContainerHijacker test case below.

TEST_CONSTEXPR_CXX14 bool test_arrays_and_initializer_lists_forward()
{
  {
    int a[] = {1, 2, 3};
    ASSERT_SAME_TYPE(decltype(std::begin(a)), int*);
    ASSERT_SAME_TYPE(decltype(std::end(a)), int*);
    assert(std::begin(a) == a);
    assert(std::end(a) == a + 3);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(decltype(std::cbegin(a)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(a)), const int*);
    assert(std::cbegin(a) == a);
    assert(std::cend(a) == a + 3);
#endif

    const auto& ca = a;
    ASSERT_SAME_TYPE(decltype(std::begin(ca)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(ca)), const int*);
    assert(std::begin(ca) == a);
    assert(std::end(ca) == a + 3);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(decltype(std::cbegin(ca)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(ca)), const int*);
    assert(std::cbegin(ca) == a);
    assert(std::cend(ca) == a + 3);
#endif
  }
  {
    std::initializer_list<int> il = {1, 2, 3};
    ASSERT_SAME_TYPE(decltype(std::begin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(il)), const int*);
    assert(std::begin(il) == il.begin());
    assert(std::end(il) == il.end());
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(decltype(std::cbegin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(il)), const int*);
    assert(std::cbegin(il) == il.begin());
    assert(std::cend(il) == il.end());
#endif

    const auto& cil = il;
    ASSERT_SAME_TYPE(decltype(std::begin(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(cil)), const int*);
    assert(std::begin(cil) == il.begin());
    assert(std::end(cil) == il.end());
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(decltype(std::cbegin(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(cil)), const int*);
    assert(std::cbegin(cil) == il.begin());
    assert(std::cend(cil) == il.end());
#endif
  }
  return true;
}

#if TEST_STD_VER > 11
TEST_CONSTEXPR_CXX17 bool test_arrays_and_initializer_lists_backward()
{
  {
    int a[] = {1, 2, 3};
    ASSERT_SAME_TYPE(decltype(std::rbegin(a)), std::reverse_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(a)), std::reverse_iterator<int*>);
    assert(std::rbegin(a).base() == a + 3);
    assert(std::rend(a).base() == a);
    ASSERT_SAME_TYPE(decltype(std::crbegin(a)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(a)), std::reverse_iterator<const int*>);
    assert(std::crbegin(a).base() == a + 3);
    assert(std::crend(a).base() == a);

    const auto& ca = a;
    ASSERT_SAME_TYPE(decltype(std::rbegin(ca)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(ca)), std::reverse_iterator<const int*>);
    assert(std::rbegin(ca).base() == a + 3);
    assert(std::rend(ca).base() == a);
    ASSERT_SAME_TYPE(decltype(std::crbegin(ca)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(ca)), std::reverse_iterator<const int*>);
    assert(std::crbegin(ca).base() == a + 3);
    assert(std::crend(ca).base() == a);
  }
  {
    std::initializer_list<int> il = {1, 2, 3};
    ASSERT_SAME_TYPE(decltype(std::rbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(il)), std::reverse_iterator<const int*>);
    assert(std::rbegin(il).base() == il.end());
    assert(std::rend(il).base() == il.begin());
    ASSERT_SAME_TYPE(decltype(std::crbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(il)), std::reverse_iterator<const int*>);
    assert(std::crbegin(il).base() == il.end());
    assert(std::crend(il).base() == il.begin());

    const auto& cil = il;
    ASSERT_SAME_TYPE(decltype(std::rbegin(cil)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(cil)), std::reverse_iterator<const int*>);
    assert(std::rbegin(cil).base() == il.end());
    assert(std::rend(cil).base() == il.begin());
    ASSERT_SAME_TYPE(decltype(std::crbegin(cil)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(cil)), std::reverse_iterator<const int*>);
    assert(std::crbegin(cil).base() == il.end());
    assert(std::crend(cil).base() == il.begin());
  }
  return true;
}
#endif

template<typename C>
TEST_CONSTEXPR_CXX14 bool test_container() {
  C c = {1, 2, 3};
  ASSERT_SAME_TYPE(decltype(std::begin(c)), typename C::iterator);
  ASSERT_SAME_TYPE(decltype(std::end(c)), typename C::iterator);
  assert(std::begin(c) == c.begin());
  assert(std::end(c) == c.end());
#if TEST_STD_VER > 11
  ASSERT_SAME_TYPE(decltype(std::cbegin(c)), typename C::const_iterator);
  ASSERT_SAME_TYPE(decltype(std::cend(c)), typename C::const_iterator);
  assert(std::cbegin(c) == c.begin());
  assert(std::cend(c) == c.end());
  ASSERT_SAME_TYPE(decltype(std::rbegin(c)), typename C::reverse_iterator);
  ASSERT_SAME_TYPE(decltype(std::rend(c)), typename C::reverse_iterator);
  assert(std::rbegin(c).base() == c.end());
  assert(std::rend(c).base() == c.begin());
  ASSERT_SAME_TYPE(decltype(std::crbegin(c)), typename C::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(std::crend(c)), typename C::const_reverse_iterator);
  assert(std::crbegin(c).base() == c.end());
  assert(std::crend(c).base() == c.begin());
#endif

  const C& cc = c;
  ASSERT_SAME_TYPE(decltype(std::begin(cc)), typename C::const_iterator);
  ASSERT_SAME_TYPE(decltype(std::end(cc)), typename C::const_iterator);
  assert(std::begin(cc) == c.begin());
  assert(std::end(cc) == c.end());
#if TEST_STD_VER > 11
  ASSERT_SAME_TYPE(decltype(std::cbegin(cc)), typename C::const_iterator);
  ASSERT_SAME_TYPE(decltype(std::cend(cc)), typename C::const_iterator);
  assert(std::cbegin(cc) == c.begin());
  assert(std::cend(cc) == c.end());
  ASSERT_SAME_TYPE(decltype(std::rbegin(cc)), typename C::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(std::rend(cc)), typename C::const_reverse_iterator);
  assert(std::rbegin(cc).base() == c.end());
  assert(std::rend(cc).base() == c.begin());
  ASSERT_SAME_TYPE(decltype(std::crbegin(cc)), typename C::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(std::crend(cc)), typename C::const_reverse_iterator);
  assert(std::crbegin(cc).base() == c.end());
  assert(std::crend(cc).base() == c.begin());
#endif

  return true;
}

struct ArrayHijacker {
  friend constexpr int begin(ArrayHijacker(&)[3]) { return 42; }
  friend constexpr int end(ArrayHijacker(&)[3]) { return 42; }
  friend constexpr int begin(const ArrayHijacker(&)[3]) { return 42; }
  friend constexpr int end(const ArrayHijacker(&)[3]) { return 42; }
};

struct ContainerHijacker {
  int *a_;
  constexpr int *begin() const { return a_; }
  constexpr int *end() const { return a_ + 3; }
  constexpr int *rbegin() const { return a_; }
  constexpr int *rend() const { return a_ + 3; }
  friend constexpr int begin(ContainerHijacker&) { return 42; }
  friend constexpr int end(ContainerHijacker&) { return 42; }
  friend constexpr int begin(const ContainerHijacker&) { return 42; }
  friend constexpr int end(const ContainerHijacker&) { return 42; }
  friend constexpr int cbegin(ContainerHijacker&) { return 42; }
  friend constexpr int cend(ContainerHijacker&) { return 42; }
  friend constexpr int cbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int cend(const ContainerHijacker&) { return 42; }
  friend constexpr int rbegin(ContainerHijacker&) { return 42; }
  friend constexpr int rend(ContainerHijacker&) { return 42; }
  friend constexpr int rbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int rend(const ContainerHijacker&) { return 42; }
  friend constexpr int crbegin(ContainerHijacker&) { return 42; }
  friend constexpr int crend(ContainerHijacker&) { return 42; }
  friend constexpr int crbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int crend(const ContainerHijacker&) { return 42; }
};

TEST_CONSTEXPR_CXX17 bool test_adl_proofing() {
  // https://llvm.org/PR28927
  {
    ArrayHijacker a[3] = {};
    assert(begin(a) == 42);
    assert(end(a) == 42);
    assert(std::begin(a) == a);
    assert(std::end(a) == a + 3);
#if TEST_STD_VER > 11
    assert(std::cbegin(a) == a);
    assert(std::cend(a) == a + 3);
    assert(std::rbegin(a).base() == a + 3);
    assert(std::rend(a).base() == a);
    assert(std::crbegin(a).base() == a + 3);
    assert(std::crend(a).base() == a);
#endif
  }
  {
    int a[3] = {};
    ContainerHijacker c{a};
    assert(begin(c) == 42);
    assert(end(c) == 42);
    assert(std::begin(c) == a);
    assert(std::end(c) == a + 3);
#if TEST_STD_VER > 11
    assert(std::cbegin(c) == a);
    assert(std::cend(c) == a + 3);
    assert(std::rbegin(c) == a);
    assert(std::rend(c) == a + 3);
    assert(std::crbegin(c) == a);
    assert(std::crend(c) == a + 3);
#endif
  }
  return true;
}

int main(int, char**) {
  test_arrays_and_initializer_lists_forward();
#if TEST_STD_VER > 11
  test_arrays_and_initializer_lists_backward();
#endif
  test_container<std::array<int, 3>>();
  test_container<std::list<int>>();
  test_container<std::vector<int>>();
  test_adl_proofing();

#if TEST_STD_VER > 11
  static_assert(test_arrays_and_initializer_lists_forward(), "");
#endif
#if TEST_STD_VER > 14
  static_assert(test_arrays_and_initializer_lists_backward());
  static_assert(test_container<std::array<int, 3>>());
  static_assert(test_adl_proofing());
#endif

  return 0;
}
