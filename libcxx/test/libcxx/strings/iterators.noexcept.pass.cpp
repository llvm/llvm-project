//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __libcpp_is_trivial_iterator<Tp>

// __libcpp_string_gets_noexcept_iterator determines if an iterator can be used
// w/o worrying about whether or not certain operations can throw.
// This gives us a "fast path for string operations".
//
// When exceptions are disabled, all iterators should get this "fast path"
//

// ADDITIONAL_COMPILE_FLAGS: -fno-exceptions

#include <iterator>
#include <cassert>
#include <string>
#include <vector>
#include <initializer_list>

#include "test_macros.h"
#include "test_iterators.h"

int main(int, char**)
{
//  basic tests
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<char *>::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<const char *>::value), "");

    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::move_iterator<char *> >      ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::move_iterator<const char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::reverse_iterator<char *> >      ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::reverse_iterator<const char *> >::value), "");

    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::__wrap_iter<char *> >      ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::__wrap_iter<const char *> >::value), "");

    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::reverse_iterator<std::__wrap_iter<char *> > > ::value), "");

//  iterators in the libc++ test suite
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<output_iterator       <char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<input_iterator        <char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<forward_iterator      <char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<bidirectional_iterator<char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<random_access_iterator<char *> >::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<ThrowingIterator      <char *> >::value), "");

    static_assert(( std::__libcpp_string_gets_noexcept_iterator<NonThrowingIterator   <char *> >::value), "");

//
//  iterators from libc++'s containers
//

//  string
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::vector<char>::iterator>              ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::vector<char>::const_iterator>        ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::vector<char>::reverse_iterator>      ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::vector<char>::const_reverse_iterator>::value), "");

//  vector
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::basic_string<char>::iterator>              ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::basic_string<char>::const_iterator>        ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::basic_string<char>::reverse_iterator>      ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::basic_string<char>::const_reverse_iterator>::value), "");

#if TEST_STD_VER >= 11
//  Initializer list  (which has no reverse iterators)
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::initializer_list<char>::iterator>              ::value), "");
    static_assert(( std::__libcpp_string_gets_noexcept_iterator<std::initializer_list<char>::const_iterator>        ::value), "");
#endif

  return 0;
}
