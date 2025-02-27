//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const basic_string<charT,traits,Allocator>& str,
//              size_type pos, size_type n,
//              const Allocator& a = Allocator()); // constexpr since C++20
//
// basic_string(const basic_string<charT,traits,Allocator>& str,
//              size_type pos,
//              const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <scoped_allocator>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S str, unsigned pos) {
  typedef typename S::traits_type T;
  typedef typename S::allocator_type A;

  if (pos <= str.size()) {
    S s2(str, pos);
    LIBCPP_ASSERT(s2.__invariants());
    typename S::size_type rlen = str.size() - pos;
    assert(s2.size() == rlen);
    assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
    LIBCPP_ASSERT(is_string_asan_correct(s2));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  else if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      S s2(str, pos);
      assert(false);
    } catch (std::out_of_range&) {
      assert(pos > str.size());
    }
  }
#endif
}

template <class S>
TEST_CONSTEXPR_CXX20 void test(S str, unsigned pos, unsigned n) {
  typedef typename S::traits_type T;
  typedef typename S::allocator_type A;
  if (pos <= str.size()) {
    S s2(str, pos, n);
    LIBCPP_ASSERT(s2.__invariants());
    typename S::size_type rlen = std::min<typename S::size_type>(str.size() - pos, n);
    assert(s2.size() == rlen);
    assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
    LIBCPP_ASSERT(is_string_asan_correct(s2));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  else if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      S s2(str, pos, n);
      assert(false);
    } catch (std::out_of_range&) {
      assert(pos > str.size());
    }
  }
#endif
}

template <class S>
TEST_CONSTEXPR_CXX20 void test(S str, unsigned pos, unsigned n, const typename S::allocator_type& a) {
  typedef typename S::traits_type T;

  if (pos <= str.size()) {
    S s2(str, pos, n, a);
    LIBCPP_ASSERT(s2.__invariants());
    typename S::size_type rlen = std::min<typename S::size_type>(str.size() - pos, n);
    assert(s2.size() == rlen);
    assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
    LIBCPP_ASSERT(is_string_asan_correct(s2));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  else if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      S s2(str, pos, n, a);
      assert(false);
    } catch (std::out_of_range&) {
      assert(pos > str.size());
    }
  }
#endif
}

void test_lwg2583() {
#if TEST_STD_VER >= 11 && !defined(TEST_HAS_NO_EXCEPTIONS)
  typedef std::basic_string<char, std::char_traits<char>, test_allocator<char> > StringA;
  std::vector<StringA, std::scoped_allocator_adaptor<test_allocator<StringA> > > vs;
  StringA s{"1234"};
  vs.emplace_back(s, 2);

  try {
    vs.emplace_back(s, 5);
  } catch (const std::out_of_range&) {
    return;
  }
  assert(false);
#endif
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a1, const Alloc& a2) {
  using S = std::basic_string<char, std::char_traits<char>, Alloc>;

  test(S(Alloc(a1)), 0);
  test(S(Alloc(a1)), 1);
  test(S("1", Alloc(a1)), 0);
  test(S("1", Alloc(a1)), 1);
  test(S("1", Alloc(a1)), 2);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 0);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 5);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 500);

  test(S(Alloc(a1)), 0, 0);
  test(S(Alloc(a1)), 0, 1);
  test(S(Alloc(a1)), 1, 0);
  test(S(Alloc(a1)), 1, 1);
  test(S(Alloc(a1)), 1, 2);
  test(S("1", Alloc(a1)), 0, 0);
  test(S("1", Alloc(a1)), 0, 1);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 0);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 1);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 10);
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 100);

  test(S(Alloc(a1)), 0, 0, Alloc(a2));
  test(S(Alloc(a1)), 0, 1, Alloc(a2));
  test(S(Alloc(a1)), 1, 0, Alloc(a2));
  test(S(Alloc(a1)), 1, 1, Alloc(a2));
  test(S(Alloc(a1)), 1, 2, Alloc(a2));
  test(S("1", Alloc(a1)), 0, 0, Alloc(a2));
  test(S("1", Alloc(a1)), 0, 1, Alloc(a2));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 0, Alloc(a2));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 1, Alloc(a2));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 10, Alloc(a2));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", Alloc(a1)), 50, 100, Alloc(a2));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>(), std::allocator<char>());
  test_string(test_allocator<char>(), test_allocator<char>());
  test_string(test_allocator<char>(3), test_allocator<char>(5));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>(), min_allocator<char>());
  test_string(safe_allocator<char>(), safe_allocator<char>());
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif
  test_lwg2583();

  return 0;
}
