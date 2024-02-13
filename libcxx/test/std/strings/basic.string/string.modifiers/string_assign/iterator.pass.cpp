//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   basic_string& assign(InputIterator first, InputIterator last); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S, class It>
TEST_CONSTEXPR_CXX20 void test(S s, It first, It last, S expected) {
  s.assign(first, last);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Widget {
  operator char() const { throw 42; }
};

template <class S, class It>
void test_exceptions(S s, It first, It last) {
  S original                 = s;
  typename S::iterator begin = s.begin();
  typename S::iterator end   = s.end();

  try {
    s.assign(first, last);
    assert(false);
  } catch (...) {
  }

  // Part of "no effects" is that iterators and pointers
  // into the string must not have been invalidated.
  LIBCPP_ASSERT(s.__invariants());
  assert(s == original);
  assert(s.begin() == begin);
  assert(s.end() == end);
}
#endif

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  {
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), s, s, S());
    test(S(), s, s + 1, S("A"));
    test(S(), s, s + 10, S("ABCDEFGHIJ"));
    test(S(), s, s + 52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), s, s, S());
    test(S("12345"), s, s + 1, S("A"));
    test(S("12345"), s, s + 10, S("ABCDEFGHIJ"));
    test(S("12345"), s, s + 52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), s, s, S());
    test(S("1234567890"), s, s + 1, S("A"));
    test(S("1234567890"), s, s + 10, S("ABCDEFGHIJ"));
    test(S("1234567890"), s, s + 52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), s, s, S());
    test(S("12345678901234567890"), s, s + 1, S("A"));
    test(S("12345678901234567890"), s, s + 10, S("ABCDEFGHIJ"));
    test(S("12345678901234567890"), s, s + 52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1), S("A"));
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), S("ABCDEFGHIJ"));
    test(S(),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1), S("A"));
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), S("ABCDEFGHIJ"));
    test(S("12345"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1), S("A"));
    test(S("1234567890"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 10),
         S("ABCDEFGHIJ"));
    test(S("1234567890"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S("12345678901234567890"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 1),
         S("A"));
    test(S("12345678901234567890"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 10),
         S("ABCDEFGHIJ"));
    test(S("12345678901234567890"),
         cpp17_input_iterator<const char*>(s),
         cpp17_input_iterator<const char*>(s + 52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) { // test iterator operations that throw
    typedef ThrowingIterator<char> TIter;
    typedef cpp17_input_iterator<TIter> IIter;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test_exceptions(S(), IIter(TIter(s, s + 10, 4, TIter::TAIncrement)), IIter(TIter()));
    test_exceptions(S(), IIter(TIter(s, s + 10, 5, TIter::TADereference)), IIter(TIter()));
    test_exceptions(S(), IIter(TIter(s, s + 10, 6, TIter::TAComparison)), IIter(TIter()));

    test_exceptions(S(), TIter(s, s + 10, 4, TIter::TAIncrement), TIter());
    test_exceptions(S(), TIter(s, s + 10, 5, TIter::TADereference), TIter());
    test_exceptions(S(), TIter(s, s + 10, 6, TIter::TAComparison), TIter());

    Widget w[100];
    test_exceptions(S(), w, w + 100);
  }
#endif

  { // test assigning to self
    S s_short = "123/";
    S s_long  = "Lorem ipsum dolor sit amet, consectetur/";

    s_short.assign(s_short.begin(), s_short.end());
    assert(s_short == "123/");
    s_short.assign(s_short.begin() + 2, s_short.end());
    assert(s_short == "3/");

    s_long.assign(s_long.begin(), s_long.end());
    assert(s_long == "Lorem ipsum dolor sit amet, consectetur/");

    s_long.assign(s_long.begin() + 30, s_long.end());
    assert(s_long == "nsectetur/");
  }

  { // test assigning a different type
    const std::uint8_t p[] = "ABCD";

    S s;
    s.assign(p, p + 4);
    assert(s == "ABCD");
  }

  { // regression-test assigning to self in sneaky ways
    S sneaky = "hello";
    sneaky.resize(sneaky.capacity(), 'x');
    S expected = sneaky + S(1, '\0');
    test(sneaky, sneaky.data(), sneaky.data() + sneaky.size() + 1, expected);
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char> > >();
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
