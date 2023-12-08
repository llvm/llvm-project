//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(size_type n, charT c, const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Alloc, class charT>
TEST_CONSTEXPR_CXX20 void test(unsigned n, charT c) {
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(n, c);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == n);
  for (unsigned i = 0; i < n; ++i)
    assert(s2[i] == c);
  assert(s2.get_allocator() == Alloc());
  assert(s2.capacity() >= s2.size());
}

template <class Alloc, class charT>
TEST_CONSTEXPR_CXX20 void test(unsigned n, charT c, const Alloc& a) {
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(n, c, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == n);
  for (unsigned i = 0; i < n; ++i)
    assert(s2[i] == c);
  assert(s2.get_allocator() == a);
  assert(s2.capacity() >= s2.size());
}

template <class Alloc, class Tp>
TEST_CONSTEXPR_CXX20 void test(Tp n, Tp c) {
  typedef char charT;
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(n, c);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    assert(s2[i] == c);
  assert(s2.get_allocator() == Alloc());
  assert(s2.capacity() >= s2.size());
}

template <class Alloc, class Tp>
TEST_CONSTEXPR_CXX20 void test(Tp n, Tp c, const Alloc& a) {
  typedef char charT;
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(n, c, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    assert(s2[i] == c);
  assert(s2.get_allocator() == a);
  assert(s2.capacity() >= s2.size());
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  test<Alloc>(0, 'a');
  test<Alloc>(0, 'a', Alloc(a));

  test<Alloc>(1, 'a');
  test<Alloc>(1, 'a', Alloc(a));

  test<Alloc>(10, 'a');
  test<Alloc>(10, 'a', Alloc(a));

  test<Alloc>(100, 'a');
  test<Alloc>(100, 'a', Alloc(a));

  test<Alloc>(static_cast<char>(100), static_cast<char>(65));
  test<Alloc>(static_cast<char>(100), static_cast<char>(65), a);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>());
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(2));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>());
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
