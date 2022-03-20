//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   basic_string& assign(InputIterator first, InputIterator last);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class S, class It>
TEST_CONSTEXPR_CXX20 void
test(S s, It first, It last, S expected)
{
    s.assign(first, last);
    LIBCPP_ASSERT(s.__invariants());
    assert(s == expected);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Widget { operator char() const { throw 42; } };

template <class S, class It>
void
test_exceptions(S s, It first, It last)
{
    S original = s;
    typename S::iterator begin = s.begin();
    typename S::iterator end = s.end();

    try {
        s.assign(first, last);
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

bool test() {
  {
    typedef std::string S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), s, s, S());
    test(S(), s, s+1, S("A"));
    test(S(), s, s+10, S("ABCDEFGHIJ"));
    test(S(), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), s, s, S());
    test(S("12345"), s, s+1, S("A"));
    test(S("12345"), s, s+10, S("ABCDEFGHIJ"));
    test(S("12345"), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), s, s, S());
    test(S("1234567890"), s, s+1, S("A"));
    test(S("1234567890"), s, s+10, S("ABCDEFGHIJ"));
    test(S("1234567890"), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), s, s, S());
    test(S("12345678901234567890"), s, s+1, S("A"));
    test(S("12345678901234567890"), s, s+10, S("ABCDEFGHIJ"));
    test(S("12345678901234567890"), s, s+52,
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("A"));
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), s, s, S());
    test(S(), s, s+1, S("A"));
    test(S(), s, s+10, S("ABCDEFGHIJ"));
    test(S(), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), s, s, S());
    test(S("12345"), s, s+1, S("A"));
    test(S("12345"), s, s+10, S("ABCDEFGHIJ"));
    test(S("12345"), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), s, s, S());
    test(S("1234567890"), s, s+1, S("A"));
    test(S("1234567890"), s, s+10, S("ABCDEFGHIJ"));
    test(S("1234567890"), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), s, s, S());
    test(S("12345678901234567890"), s, s+1, S("A"));
    test(S("12345678901234567890"), s, s+10, S("ABCDEFGHIJ"));
    test(S("12345678901234567890"), s, s+52,
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), S());
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1), S("A"));
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S(), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("12345"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("1234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s),
         S());
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+1),
         S("A"));
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S("12345678901234567890"), cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
  }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) { // test iterator operations that throw
    typedef std::string S;
    typedef ThrowingIterator<char> TIter;
    typedef cpp17_input_iterator<TIter> IIter;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test_exceptions(S(), IIter(TIter(s, s+10, 4, TIter::TAIncrement)), IIter(TIter()));
    test_exceptions(S(), IIter(TIter(s, s+10, 5, TIter::TADereference)), IIter(TIter()));
    test_exceptions(S(), IIter(TIter(s, s+10, 6, TIter::TAComparison)), IIter(TIter()));

    test_exceptions(S(), TIter(s, s+10, 4, TIter::TAIncrement), TIter());
    test_exceptions(S(), TIter(s, s+10, 5, TIter::TADereference), TIter());
    test_exceptions(S(), TIter(s, s+10, 6, TIter::TAComparison), TIter());

    Widget w[100];
    test_exceptions(S(), w, w+100);
  }
#endif

  { // test assigning to self
    typedef std::string S;
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
    typedef std::string S;
    const uint8_t p[] = "ABCD";

    S s;
    s.assign(p, p + 4);
    assert(s == "ABCD");
  }

  { // regression-test assigning to self in sneaky ways
    std::string sneaky = "hello";
    sneaky.resize(sneaky.capacity(), 'x');
    std::string expected = sneaky + std::string(1, '\0');
    test(sneaky, sneaky.data(), sneaky.data() + sneaky.size() + 1, expected);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

    return 0;
}
