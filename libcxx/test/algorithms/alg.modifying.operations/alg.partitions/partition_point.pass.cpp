//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator, class Predicate>
//     ForwardIterator
//     partition_point(ForwardIterator first, ForwardIterator last, Predicate pred);

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
        const int ia[] = {2, 4, 6, 8, 10};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia));
    }
    {
        const int ia[] = {1, 2, 4, 6, 8};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 1));
    }
    {
        const int ia[] = {1, 3, 2, 4, 6};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 2));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 3));
    }
    {
        const int ia[] = {1, 3, 5, 7, 2, 4};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 4));
    }
    {
        const int ia[] = {1, 3, 5, 7, 9, 2};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 5));
    }
    {
        const int ia[] = {1, 3, 5, 7, 9, 11};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::end(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia + 6));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6, 7};
        assert(std::partition_point(forward_iterator<const int*>(std::begin(ia)),
                                    forward_iterator<const int*>(std::begin(ia)),
                                    is_odd()) == forward_iterator<const int*>(ia));
    }
}
