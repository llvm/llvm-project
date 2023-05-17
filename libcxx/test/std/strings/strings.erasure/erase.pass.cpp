//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// template <class charT, class traits, class Allocator, class U>
//   typename basic_string<charT, traits, Allocator>::size_type
//   erase(basic_string<charT, traits, Allocator>& c, const U& value);

#include <string>
#include <optional>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S, class U>
void test0(S s, U val, S expected, std::size_t expected_erased_count) {
  ASSERT_SAME_TYPE(typename S::size_type, decltype(std::erase(s, val)));
  assert(expected_erased_count == std::erase(s, val));
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
}

template <class S>
void test()
{

  test0(S(""), 'a', S(""), 0);

  test0(S("a"), 'a', S(""), 1);
  test0(S("a"), 'b', S("a"), 0);

  test0(S("ab"), 'a', S("b"), 1);
  test0(S("ab"), 'b', S("a"), 1);
  test0(S("ab"), 'c', S("ab"), 0);
  test0(S("aa"), 'a', S(""), 2);
  test0(S("aa"), 'c', S("aa"), 0);

  test0(S("abc"), 'a', S("bc"), 1);
  test0(S("abc"), 'b', S("ac"), 1);
  test0(S("abc"), 'c', S("ab"), 1);
  test0(S("abc"), 'd', S("abc"), 0);

  test0(S("aab"), 'a', S("b"), 2);
  test0(S("aab"), 'b', S("aa"), 1);
  test0(S("aab"), 'c', S("aab"), 0);
  test0(S("abb"), 'a', S("bb"), 1);
  test0(S("abb"), 'b', S("a"), 2);
  test0(S("abb"), 'c', S("abb"), 0);
  test0(S("aaa"), 'a', S(""), 3);
  test0(S("aaa"), 'b', S("aaa"), 0);

  //  Test cross-type erasure
  using opt = std::optional<typename S::value_type>;
  test0(S("aba"), opt(), S("aba"), 0);
  test0(S("aba"), opt('a'), S("b"), 2);
  test0(S("aba"), opt('b'), S("aa"), 1);
  test0(S("aba"), opt('c'), S("aba"), 0);
}

int main(int, char**)
{
    test<std::string>();
    test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>> ();
    test<std::basic_string<char, std::char_traits<char>, test_allocator<char>>> ();

  return 0;
}
