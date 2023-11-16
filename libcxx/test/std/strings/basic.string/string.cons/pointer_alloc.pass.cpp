//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const charT* s, const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class Alloc, class charT>
TEST_CONSTEXPR_CXX20 void test(const charT* s) {
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  typedef typename S::traits_type T;
  std::size_t n = T::length(s);
  S s2(s);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == n);
  assert(T::compare(s2.data(), s, n) == 0);
  assert(s2.get_allocator() == Alloc());
  assert(s2.capacity() >= s2.size());
}

template <class Alloc, class charT>
TEST_CONSTEXPR_CXX20 void test(const charT* s, const Alloc& a) {
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  typedef typename S::traits_type T;
  std::size_t n = T::length(s);
  S s2(s, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == n);
  assert(T::compare(s2.data(), s, n) == 0);
  assert(s2.get_allocator() == a);
  assert(s2.capacity() >= s2.size());
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test(const Alloc& a) {
  test<Alloc>("");
  test<Alloc>("", Alloc(a));

  test<Alloc>("1");
  test<Alloc>("1", Alloc(a));

  test<Alloc>("1234567980");
  test<Alloc>("1234567980", Alloc(a));

  test<Alloc>("123456798012345679801234567980123456798012345679801234567980");
  test<Alloc>("123456798012345679801234567980123456798012345679801234567980", Alloc(a));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test(std::allocator<char>());
  test(test_allocator<char>());
  test(test_allocator<char>(2));
#if TEST_STD_VER >= 11
  test(min_allocator<char>());
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
