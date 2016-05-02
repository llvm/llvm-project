//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   iterator insert(const_iterator p, InputIterator first, InputIterator last);

#if _LIBCPP_DEBUG >= 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <string>
#include <cassert>

#include "test_iterators.h"
#include "min_allocator.h"

template <class S, class It>
void
test(S s, typename S::difference_type pos, It first, It last, S expected)
{
    typename S::const_iterator p = s.cbegin() + pos;
    typename S::iterator i = s.insert(p, first, last);
    LIBCPP_ASSERT(s.__invariants());
    assert(i - s.begin() == pos);
    assert(s == expected);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class S, class It>
void
test_exceptions(S s, typename S::difference_type pos, It first, It last)
{
    typename S::const_iterator p = s.cbegin() + pos;
	S aCopy = s;
    try {
        s.insert(p, first, last);
        assert(false);
        }
    catch (...) {}
    LIBCPP_ASSERT(s.__invariants());
    assert(s == aCopy);
}
#endif

int main()
{
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

    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S());
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("A"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("ABCDEFGHIJ"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345"));
    test(S("12345"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A2345"));
    test(S("12345"), 4, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("1234567890"));
    test(S("1234567890"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A234567890"));
    test(S("1234567890"), 10, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345678901234567890"));
    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, input_iterator<const char*>(s), input_iterator<const char*>(s+52),
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

    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S());
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("A"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("ABCDEFGHIJ"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345"));
    test(S("12345"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A2345"));
    test(S("12345"), 4, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("1234567890"));
    test(S("1234567890"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A234567890"));
    test(S("1234567890"), 10, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345678901234567890"));
    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
	{ // test iterator operations that throw
    typedef std::string S;
    typedef ThrowingIterator<char> TIter;
    typedef input_iterator<TIter> IIter;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 4, TIter::TAIncrement)), IIter());
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 5, TIter::TADereference)), IIter());
    test_exceptions(S(), 0, IIter(TIter(s, s+10, 6, TIter::TAComparison)), IIter());

    test_exceptions(S(), 0, TIter(s, s+10, 4, TIter::TAIncrement), TIter());
    test_exceptions(S(), 0, TIter(s, s+10, 5, TIter::TADereference), TIter());
    test_exceptions(S(), 0, TIter(s, s+10, 6, TIter::TAComparison), TIter());
	}
#endif
#if _LIBCPP_DEBUG >= 1
    {
        std::string v;
        std::string v2;
        char a[] = "123";
        const int N = sizeof(a)/sizeof(a[0]);
        std::string::iterator i = v.insert(v2.cbegin() + 10, a, a+N);
        assert(false);
    }
#endif
}
