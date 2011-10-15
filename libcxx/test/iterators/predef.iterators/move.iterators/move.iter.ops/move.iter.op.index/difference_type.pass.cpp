//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// requires RandomAccessIterator<Iter>
//   unspecified operator[](difference_type n) const;

#include <iterator>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
#include <memory>
#endif

#include "../../../../iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n,
     typename std::iterator_traits<It>::value_type x)
{
    typedef typename std::iterator_traits<It>::value_type value_type;
    const std::move_iterator<It> r(i);
    value_type rr = r[n];
    assert(rr == x);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

struct do_nothing
{
    void operator()(void*) const {}
};

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    char s[] = "1234567890";
    test(random_access_iterator<char*>(s+5), 4, '0');
    test(s+5, 4, '0');
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    int i[5];
    typedef std::unique_ptr<int, do_nothing> Ptr;
    Ptr p[5];
    for (unsigned j = 0; j < 5; ++j)
        p[j].reset(i+j);
    test(p, 3, Ptr(i+3));
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
