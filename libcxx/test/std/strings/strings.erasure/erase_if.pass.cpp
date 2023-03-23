//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// template <class charT, class traits, class Allocator, class Predicate>
//   typename basic_string<charT, traits, Allocator>::size_type
//   erase_if(basic_string<charT, traits, Allocator>& c, Predicate pred);

#include <string>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S, class Pred>
void test0(S s, Pred p, S expected, std::size_t expected_erased_count) {
  ASSERT_SAME_TYPE(typename S::size_type, decltype(std::erase_if(s, p)));
  assert(expected_erased_count == std::erase_if(s, p));
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
}

template <typename S>
void test()
{
    auto isA = [](auto ch) { return ch == 'a';};
    auto isB = [](auto ch) { return ch == 'b';};
    auto isC = [](auto ch) { return ch == 'c';};
    auto isD = [](auto ch) { return ch == 'd';};
    auto True  = [](auto) { return true; };
    auto False = [](auto) { return false; };

    test0(S(""), isA, S(""), 0);

    test0(S("a"), isA, S(""), 1);
    test0(S("a"), isB, S("a"), 0);

    test0(S("ab"), isA, S("b"), 1);
    test0(S("ab"), isB, S("a"), 1);
    test0(S("ab"), isC, S("ab"), 0);
    test0(S("aa"), isA, S(""), 2);
    test0(S("aa"), isC, S("aa"), 0);

    test0(S("abc"), isA, S("bc"), 1);
    test0(S("abc"), isB, S("ac"), 1);
    test0(S("abc"), isC, S("ab"), 1);
    test0(S("abc"), isD, S("abc"), 0);

    test0(S("aab"), isA, S("b"), 2);
    test0(S("aab"), isB, S("aa"), 1);
    test0(S("aab"), isC, S("aab"), 0);
    test0(S("abb"), isA, S("bb"), 1);
    test0(S("abb"), isB, S("a"), 2);
    test0(S("abb"), isC, S("abb"), 0);
    test0(S("aaa"), isA, S(""), 3);
    test0(S("aaa"), isB, S("aaa"), 0);

    test0(S("aba"), False, S("aba"), 0);
    test0(S("aba"), True, S(""), 3);
}

int main(int, char**)
{
    test<std::string>();
    test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>> ();
    test<std::basic_string<char, std::char_traits<char>, test_allocator<char>>> ();

  return 0;
}
