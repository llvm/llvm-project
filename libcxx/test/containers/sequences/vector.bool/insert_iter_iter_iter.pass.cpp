//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// template <class Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

#include <vector>
#include <cassert>
#include "test_iterators.h"

int main()
{
    {
        std::vector<bool> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, input_iterator<const bool*>(a),
                                        input_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (int k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < 105; ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const bool*>(a),
                                        forward_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (int k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < 105; ++j)
            assert(v[j] == 0);
    }
}
