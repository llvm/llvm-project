//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class Predicate>
//     bool
//     is_partitioned(InputIterator first, InputIterator last, Predicate pred);

#include <algorithm>
#include <cassert>

#include "../../iterators.h"

struct is_odd
{
    bool operator()(const int& i) const {return i & 1;}
};

int main()
{
    {
        const int ia[] = {1, 2, 3, 4, 5, 6};
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    is_odd()));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6};
        assert( std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    is_odd()));
    }
    {
        const int ia[] = {2, 4, 6, 1, 3, 5};
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    is_odd()));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6, 7};
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    is_odd()));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6, 7};
        assert( std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::begin(ia)),
                                    is_odd()));
    }
}
