//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type capacity() const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

#include "test_macros.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_invariant(S s, test_allocator_statistics& alloc_stats) {
  alloc_stats.throw_after = 0;
#ifndef TEST_HAS_NO_EXCEPTIONS
  try
#endif
  {
    while (s.size() < s.capacity())
      s.push_back(typename S::value_type());
    assert(s.size() == s.capacity());
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  catch (...) {
    assert(false);
  }
#endif
  alloc_stats.throw_after = INT_MAX;
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  using S = std::basic_string<char, std::char_traits<char>, Alloc>;
  {
    S const s((Alloc(a)));
    assert(s.capacity() >= 0);
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    S const s(3, 'x', Alloc(a));
    assert(s.capacity() >= 3);
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
#if TEST_STD_VER >= 11
  // Check that we perform SSO
  {
    S const s;
    assert(s.capacity() > 0);
    ASSERT_NOEXCEPT(s.capacity());
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>());
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(3));
  test_string(min_allocator<char>());
  test_string(safe_allocator<char>());

  {
    test_allocator_statistics alloc_stats;
    typedef std::basic_string<char, std::char_traits<char>, test_allocator<char> > S;
    S s((test_allocator<char>(&alloc_stats)));
    test_invariant(s, alloc_stats);
    LIBCPP_ASSERT(is_string_asan_correct(s));
    s.assign(10, 'a');
    s.erase(5);
    test_invariant(s, alloc_stats);
    LIBCPP_ASSERT(is_string_asan_correct(s));
    s.assign(100, 'a');
    s.erase(50);
    test_invariant(s, alloc_stats);
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
