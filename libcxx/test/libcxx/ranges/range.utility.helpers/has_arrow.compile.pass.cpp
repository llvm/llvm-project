//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ranges>

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cstddef>
#include <iterator>

#include "test_macros.h"

struct simple_iter {
    typedef std::input_iterator_tag         iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;

    reference operator*() const;
    pointer operator->() const;
    friend bool operator==(const simple_iter&, const simple_iter&);
    friend bool operator< (const simple_iter&, const simple_iter&);
    friend bool operator<=(const simple_iter&, const simple_iter&);
    friend bool operator> (const simple_iter&, const simple_iter&);
    friend bool operator>=(const simple_iter&, const simple_iter&);

    simple_iter& operator++();
    simple_iter operator++(int);
};

struct no_star {
    typedef std::input_iterator_tag         iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;

 /* reference operator*() const; */
    pointer operator->() const;
    friend bool operator==(const simple_iter&, const simple_iter&);
    friend bool operator< (const simple_iter&, const simple_iter&);
    friend bool operator<=(const simple_iter&, const simple_iter&);
    friend bool operator> (const simple_iter&, const simple_iter&);
    friend bool operator>=(const simple_iter&, const simple_iter&);

    simple_iter& operator++();
    simple_iter operator++(int);
};

struct no_arrow {
    typedef std::input_iterator_tag         iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;

    reference operator*() const;
 /* pointer operator->() const; */
    friend bool operator==(const simple_iter&, const simple_iter&);
    friend bool operator< (const simple_iter&, const simple_iter&);
    friend bool operator<=(const simple_iter&, const simple_iter&);
    friend bool operator> (const simple_iter&, const simple_iter&);
    friend bool operator>=(const simple_iter&, const simple_iter&);

    simple_iter& operator++();
    simple_iter operator++(int);
};

struct E {};
struct Incomplete;

static_assert(std::__has_arrow<int*>);
static_assert(std::__has_arrow<E*>);
static_assert(!std::__has_arrow<Incomplete*>); // Because it's not an input_iterator.
static_assert(std::__has_arrow<simple_iter>);
static_assert(!std::__has_arrow<no_star>);
static_assert(!std::__has_arrow<no_arrow>);
