//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test checks that std::contiguous_iterator uses std::to_address, which is not SFINAE-friendly
// when the type is missing the `T::element_type` typedef.

#include <iterator>

#include <compare>
#include <cstddef>

struct no_element_type {
    typedef std::contiguous_iterator_tag    iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef no_element_type                 self;

    no_element_type();

    reference operator*() const;
    pointer operator->() const;
    auto operator<=>(const self&) const = default;

    self& operator++();
    self operator++(int);

    self& operator--();
    self operator--(int);

    self& operator+=(difference_type n);
    self operator+(difference_type n) const;
    friend self operator+(difference_type n, self x);

    self& operator-=(difference_type n);
    self operator-(difference_type n) const;
    difference_type operator-(const self& n) const;

    reference operator[](difference_type n) const;
};

void test() {
    (void) std::contiguous_iterator<no_element_type>;
        // expected-error@*:* {{implicit instantiation of undefined template}}
        // expected-note@*:* {{to_address}}
}
