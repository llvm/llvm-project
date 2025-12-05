//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __libcpp_is_contiguous_iterator<_Tp>

// __libcpp_is_contiguous_iterator determines if an iterator is contiguous,
// either because it advertises itself as such (in C++20) or because it
// is a pointer type or a known trivial wrapper around a pointer type,
// such as __wrap_iter<T*>.
//

#include <cassert>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

class T;  // incomplete

class my_input_iterator
{
    struct tag : std::input_iterator_tag {};
    typedef my_input_iterator Self;
    int *state_;
public:
    typedef tag iterator_category;
    typedef int value_type;
    typedef int difference_type;
    typedef int* pointer;
    typedef int& reference;

    my_input_iterator();
    reference operator*() const;
    pointer operator->() const;

    Self& operator++();
    Self operator++(int);
    friend bool operator==(const Self&, const Self&);
    friend bool operator!=(const Self&, const Self&);
};

class my_random_access_iterator
{
    struct tag : std::random_access_iterator_tag {};
    typedef my_random_access_iterator Self;
    int *state_;
public:
    typedef tag iterator_category;
    typedef int value_type;
    typedef int difference_type;
    typedef int* pointer;
    typedef int& reference;

    my_random_access_iterator();
    reference operator*() const;
    pointer operator->() const;
    reference operator[](difference_type) const;

    Self& operator++();
    Self operator++(int);
    Self& operator--();
    Self operator--(int);
    friend Self& operator+=(Self&, difference_type);
    friend Self& operator-=(Self&, difference_type);
    friend Self operator+(Self, difference_type);
    friend Self operator+(difference_type, Self);
    friend Self operator-(Self, difference_type);
    friend difference_type operator-(Self, Self);
    friend bool operator==(const Self&, const Self&);
    friend bool operator!=(const Self&, const Self&);
    friend bool operator<(const Self&, const Self&);
    friend bool operator>(const Self&, const Self&);
    friend bool operator<=(const Self&, const Self&);
    friend bool operator>=(const Self&, const Self&);
};

struct fake_deque_iterator : std::deque<int>::iterator {
    using element_type = int;
};
static_assert(std::__has_random_access_iterator_category<fake_deque_iterator>::value, "");
static_assert(!std::__libcpp_is_contiguous_iterator<fake_deque_iterator>::value, "");

int main(int, char**)
{
//  basic tests
    static_assert(( std::__libcpp_is_contiguous_iterator<char *>::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<const char *>::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<int *>::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<int **>::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<T *>::value), "");

    static_assert((!std::__libcpp_is_contiguous_iterator<my_input_iterator>::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<my_random_access_iterator>::value), "");

    // move_iterator changes value category, which makes it pretty sketchy to use in optimized codepaths
    static_assert((!std::__libcpp_is_contiguous_iterator<std::move_iterator<char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::move_iterator<const char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::move_iterator<int *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::move_iterator<T *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::move_iterator<my_random_access_iterator> >::value), "");

    static_assert((!std::__libcpp_is_contiguous_iterator<std::reverse_iterator<char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::reverse_iterator<const char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::reverse_iterator<int *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::reverse_iterator<T *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::reverse_iterator<my_random_access_iterator> >::value), "");

    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<char *> >::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<const char *> >::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<int *> >::value), "");

    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<T *> >::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<std::__wrap_iter<T *> > >::value), "");

    // Here my_random_access_iterator is standing in for some user's fancy pointer type, written pre-C++20.
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<my_random_access_iterator> >::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<std::__wrap_iter<my_random_access_iterator> > >::value), "");

//  iterators in the libc++ test suite
    static_assert((!std::__libcpp_is_contiguous_iterator<cpp17_output_iterator <char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<cpp17_input_iterator  <char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<forward_iterator      <char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<bidirectional_iterator<char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<random_access_iterator<char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<ThrowingIterator      <char *> >::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<NonThrowingIterator   <char *> >::value), "");

//
//  iterators from libc++'s containers
//

//  vector
    static_assert(( std::__libcpp_is_contiguous_iterator<std::vector<int>::iterator>                   ::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::vector<int>::const_iterator>             ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<int>::reverse_iterator>           ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<int>::const_reverse_iterator>     ::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::__wrap_iter<std::vector<int>::iterator> >::value), "");

//  string
    static_assert(( std::__libcpp_is_contiguous_iterator<std::string::iterator>              ::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::string::const_iterator>        ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::string::reverse_iterator>      ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::string::const_reverse_iterator>::value), "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    static_assert(( std::__libcpp_is_contiguous_iterator<std::wstring::iterator>              ::value), "");
    static_assert(( std::__libcpp_is_contiguous_iterator<std::wstring::const_iterator>        ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::wstring::reverse_iterator>      ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::wstring::const_reverse_iterator>::value), "");
#endif

//  deque is random-access but not contiguous
    static_assert((!std::__libcpp_is_contiguous_iterator<std::deque<int>::iterator>                   ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::deque<int>::const_iterator>             ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::deque<int>::reverse_iterator>           ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::deque<int>::const_reverse_iterator>     ::value), "");

//  vector<bool> is random-access but not contiguous
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<bool>::iterator>                   ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<bool>::const_iterator>             ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<bool>::reverse_iterator>           ::value), "");
    static_assert((!std::__libcpp_is_contiguous_iterator<std::vector<bool>::const_reverse_iterator>     ::value), "");

    return 0;
}
