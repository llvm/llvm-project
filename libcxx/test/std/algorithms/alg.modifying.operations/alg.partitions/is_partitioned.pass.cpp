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
#include <functional>
#include <cassert>

#include "test_iterators.h"
#include "counting_predicates.hpp"

struct is_odd
{
    bool operator()(const int& i) const {return i & 1;}
};

int main()
{
    {
        const int ia[] = {1, 2, 3, 4, 5, 6};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::end(ia)));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert( std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::end(ia)));
    }
    {
        const int ia[] = {2, 4, 6, 1, 3, 5};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::end(ia)));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6, 7};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert(!std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::end(ia)));
    }
    {
        const int ia[] = {1, 3, 5, 2, 4, 6, 7};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert( std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::begin(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::begin(ia)));
    }
    {
        const int ia[] = {1, 3, 5, 7, 9, 11, 2};
        unary_counting_predicate<is_odd, int> pred((is_odd()));
        assert( std::is_partitioned(input_iterator<const int*>(std::begin(ia)),
                                    input_iterator<const int*>(std::end(ia)),
                                    std::ref(pred)));
        assert(pred.count() <= std::distance(std::begin(ia), std::end(ia)));
    }
}
