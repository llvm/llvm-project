//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>
//   operator+(charT lhs, const basic_string<charT,traits,Allocator>& rhs);

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(charT lhs, basic_string<charT,traits,Allocator>&& rhs);

#include <string>
#include <utility>
#include <cassert>

#include "min_allocator.h"

template <class S>
void
test0(typename S::value_type lhs, const S& rhs, const S& x)
{
    assert(lhs + rhs == x);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

template <class S>
void
test1(typename S::value_type lhs, S&& rhs, const S& x)
{
    assert(lhs + move(rhs) == x);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    {
    typedef std::string S;
    test0('a', S(""), S("a"));
    test0('a', S("12345"), S("a12345"));
    test0('a', S("1234567890"), S("a1234567890"));
    test0('a', S("12345678901234567890"), S("a12345678901234567890"));

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    test1('a', S(""), S("a"));
    test1('a', S("12345"), S("a12345"));
    test1('a', S("1234567890"), S("a1234567890"));
    test1('a', S("12345678901234567890"), S("a12345678901234567890"));

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test0('a', S(""), S("a"));
    test0('a', S("12345"), S("a12345"));
    test0('a', S("1234567890"), S("a1234567890"));
    test0('a', S("12345678901234567890"), S("a12345678901234567890"));

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    test1('a', S(""), S("a"));
    test1('a', S("12345"), S("a12345"));
    test1('a', S("1234567890"), S("a1234567890"));
    test1('a', S("12345678901234567890"), S("a12345678901234567890"));

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    }
#endif
}
