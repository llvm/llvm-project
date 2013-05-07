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
//   bool
//   any_of(InputIterator first, InputIterator last, Predicate pred);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

struct test1
{
    bool operator()(const int& i) const
    {
        return i % 2 == 0;
    }
};

int main()
{
    {
        int ia[] = {2, 4, 6, 8};
        const unsigned sa = sizeof(ia)/sizeof(ia[0]);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia + sa), test1()) == true);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia), test1()) == false);
    }
    {
        const int ia[] = {2, 4, 5, 8};
        const unsigned sa = sizeof(ia)/sizeof(ia[0]);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia + sa), test1()) == true);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia), test1()) == false);
    }
    {
        const int ia[] = {1, 3, 5, 7};
        const unsigned sa = sizeof(ia)/sizeof(ia[0]);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia + sa), test1()) == false);
        assert(std::any_of(input_iterator<const int*>(ia),
                           input_iterator<const int*>(ia), test1()) == false);
    }
}
