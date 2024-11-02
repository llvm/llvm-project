//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// template<class InputIterator>
//   iterator insert(const_iterator p, InputIterator first, InputIterator last); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class S, class It>
TEST_CONSTEXPR_CXX20 void
test(S s, typename S::difference_type pos, It first, It last, S expected)
{
    typename S::const_iterator p = s.cbegin() + pos;
    typename S::iterator i = s.insert(p, first, last);
    LIBCPP_ASSERT(s.__invariants());
    assert(i - s.begin() == pos);
    assert(s == expected);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Widget { operator char() const { throw 42; } };

template <class S, class It>
TEST_CONSTEXPR_CXX20 void
test_exceptions(S s, typename S::difference_type pos, It first, It last)
{
    typename S::const_iterator p = s.cbegin() + pos;

    S original = s;
    typename S::iterator begin = s.begin();
    typename S::iterator end = s.end();

    try {
        s.insert(p, first, last);
        assert(false);
    } catch (...) {}

    // Part of "no effects" is that iterators and pointers
    // into the string must not have been invalidated.
    LIBCPP_ASSERT(s.__invariants());
    assert(s == original);
    assert(s.begin() == begin);
    assert(s.end() == end);
}
#endif

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), 0, s, s, S());
    test(S(), 0, s, s+1, S("A"));
    test(S(), 0, s, s+10, S("ABCDEFGHIJ"));
    test(S(), 0, s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, s, s, S("12345"));
    test(S("12345"), 1, s, s+1, S("1A2345"));
    test(S("12345"), 4, s, s+10, S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, s, s+52, S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, s, s, S("1234567890"));
    test(S("1234567890"), 1, s, s+1, S("1A234567890"));
    test(S("1234567890"), 10, s, s+10, S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, s, s+52, S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, s, s, S("12345678901234567890"));
    test(S("12345678901234567890"), 3, s, s+1, S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, s, s+10, S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, s, s+52,
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("A"));
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("ABCDEFGHIJ"));
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("12345"));
    test(S("12345"), 1, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("1A2345"));
    test(S("12345"), 4, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("1234567890"));
    test(S("1234567890"), 1, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("1A234567890"));
    test(S("1234567890"), 10, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("12345678901234567890"));
    test(S("12345678901234567890"), 3, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), 0, s, s, S());
    test(S(), 0, s, s+1, S("A"));
    test(S(), 0, s, s+10, S("ABCDEFGHIJ"));
    test(S(), 0, s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, s, s, S("12345"));
    test(S("12345"), 1, s, s+1, S("1A2345"));
    test(S("12345"), 4, s, s+10, S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, s, s+52, S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, s, s, S("1234567890"));
    test(S("1234567890"), 1, s, s+1, S("1A234567890"));
    test(S("1234567890"), 10, s, s+10, S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, s, s+52, S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, s, s, S("12345678901234567890"));
    test(S("12345678901234567890"), 3, s, s+1, S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, s, s+10, S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, s, s+52,
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("A"));
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("ABCDEFGHIJ"));
    test(S(), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("12345"));
    test(S("12345"), 1, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("1A2345"));
    test(S("12345"), 4, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("1234567890"));
    test(S("1234567890"), 1, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("1A234567890"));
    test(S("1234567890"), 10, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52), S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S("12345678901234567890"));
    test(S("12345678901234567890"), 3, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10), S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
  }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) { // test iterator operations that throw
    typedef std::string S;
    typedef ThrowingIterator<char> TIter;
    typedef cpp17_input_iterator<TIter> IIter;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 4, TIter::TAIncrement)), IIter(TIter()));
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 5, TIter::TADereference)), IIter(TIter()));
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 6, TIter::TAComparison)), IIter(TIter()));

    test_exceptions(S(), 0, TIter(s, s+10, 4, TIter::TAIncrement), TIter());
    test_exceptions(S(), 0, TIter(s, s+10, 5, TIter::TADereference), TIter());
    test_exceptions(S(), 0, TIter(s, s+10, 6, TIter::TAComparison), TIter());

    Widget w[100];
    test_exceptions(S(), 0, w, w+100);
  }
#endif

  { // test inserting into self
    typedef std::string S;
    S s_short = "123/";
    S s_long  = "Lorem ipsum dolor sit amet, consectetur/";

    s_short.insert(s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/");
    s_short.insert(s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/123/123/");
    s_short.insert(s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/123/123/123/123/123/123/");

    s_long.insert(s_long.begin(), s_long.begin(), s_long.end());
    assert(s_long == "Lorem ipsum dolor sit amet, consectetur/Lorem ipsum dolor sit amet, consectetur/");
  }

  { // test assigning a different type
    typedef std::string S;
    const uint8_t p[] = "ABCD";

    S s;
    s.insert(s.begin(), p, p + 4);
    assert(s == "ABCD");
  }

  if (!TEST_IS_CONSTANT_EVALUATED) { // regression-test inserting into self in sneaky ways
    std::string s_short = "hello";
    std::string s_long = "Lorem ipsum dolor sit amet, consectetur/";
    std::string s_othertype = "hello";
    const unsigned char *first = reinterpret_cast<const unsigned char*>(s_othertype.data());

    test(s_short, 0, s_short.data() + s_short.size(), s_short.data() + s_short.size() + 1,
         std::string("\0hello", 6));
    test(s_long, 0, s_long.data() + s_long.size(), s_long.data() + s_long.size() + 1,
         std::string("\0Lorem ipsum dolor sit amet, consectetur/", 41));
    test(s_othertype, 1, first + 2, first + 5, std::string("hlloello"));
  }

  { // test with a move iterator that returns char&&
    typedef cpp17_input_iterator<const char*> It;
    typedef std::move_iterator<It> MoveIt;
    const char p[] = "ABCD";
    std::string s;
    s.insert(s.begin(), MoveIt(It(std::begin(p))), MoveIt(It(std::end(p) - 1)));
    assert(s == "ABCD");
  }
  { // test with a move iterator that returns char&&
    typedef forward_iterator<const char*> It;
    typedef std::move_iterator<It> MoveIt;
    const char p[] = "ABCD";
    std::string s;
    s.insert(s.begin(), MoveIt(It(std::begin(p))), MoveIt(It(std::end(p) - 1)));
    assert(s == "ABCD");
  }
  { // test with a move iterator that returns char&&
    typedef const char* It;
    typedef std::move_iterator<It> MoveIt;
    const char p[] = "ABCD";
    std::string s;
    s.insert(s.begin(), MoveIt(It(std::begin(p))), MoveIt(It(std::end(p) - 1)));
    assert(s == "ABCD");
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
