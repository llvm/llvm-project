//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   basic_string(InputIterator begin, InputIterator end,
//   const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <iterator>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class Alloc, class It>
TEST_CONSTEXPR_CXX20 void test(It first, It last) {
  typedef typename std::iterator_traits<It>::value_type charT;
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(first, last);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(std::distance(first, last)));
  unsigned i = 0;
  for (It it = first; it != last;) {
    assert(s2[i] == *it);
    ++it;
    ++i;
  }
  assert(s2.get_allocator() == Alloc());
  assert(s2.capacity() >= s2.size());
  LIBCPP_ASSERT(is_string_asan_correct(s2));
}

template <class Alloc, class It>
TEST_CONSTEXPR_CXX20 void test(It first, It last, const Alloc& a) {
  typedef typename std::iterator_traits<It>::value_type charT;
  typedef std::basic_string<charT, std::char_traits<charT>, Alloc> S;
  S s2(first, last, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(std::distance(first, last)));
  unsigned i = 0;
  for (It it = first; it != last;) {
    assert(s2[i] == *it);
    ++it;
    ++i;
  }
  assert(s2.get_allocator() == a);
  assert(s2.capacity() >= s2.size());
  LIBCPP_ASSERT(is_string_asan_correct(s2));
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  const char* s = "12345678901234567890123456789012345678901234567890";

  test<Alloc>(s, s);
  test<Alloc>(s, s, Alloc(a));

  test<Alloc>(s, s + 1);
  test<Alloc>(s, s + 1, Alloc(a));

  test<Alloc>(s, s + 10);
  test<Alloc>(s, s + 10, Alloc(a));

  test<Alloc>(s, s + 50);
  test<Alloc>(s, s + 50, Alloc(a));

  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s));
  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), Alloc(a));

  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1));
  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1), Alloc(a));

  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10));
  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), Alloc(a));

  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 50));
  test<Alloc>(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 50), Alloc(a));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(2));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>());
#endif
  {
    static_assert((!std::is_constructible<std::string, std::string, std::string>::value), "");
    static_assert((!std::is_constructible<std::string, std::string, std::string, std::allocator<char> >::value), "");
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
