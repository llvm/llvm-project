//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INPUT_ITERATOR_H
#define INPUT_ITERATOR_H

#include <iterator>

template <class It>
class cpp17_input_iterator
{
    It it_;
public:
    typedef typename std::input_iterator_tag                   iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    cpp17_input_iterator() : it_() {}
    explicit cpp17_input_iterator(It it) : it_(it) {}

    reference operator*() const {return *it_;}
    pointer operator->() const {return it_;}

    cpp17_input_iterator& operator++() {++it_; return *this;}
    cpp17_input_iterator operator++(int) {cpp17_input_iterator tmp(*this); ++(*this); return tmp;}

    friend bool operator==(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
        {return x.it_ == y.it_;}
    friend bool operator!=(const cpp17_input_iterator& x, const cpp17_input_iterator& y)
        {return !(x == y);}
};

#endif // INPUT_ITERATOR_H
