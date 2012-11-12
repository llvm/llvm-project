//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template<class NotAnIterator>
// struct iterator_traits
// {
// };

#include <iterator>

struct not_an_iterator
{
};

int main()
{
    typedef std::iterator_traits<not_an_iterator> It;
}
