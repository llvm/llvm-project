//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class OutputIterator1,
//           class OutputIterator2, class Predicate>
//     pair<OutputIterator1, OutputIterator2>
//     partition_copy(InputIterator first, InputIterator last,
//                    OutputIterator1 out_true, OutputIterator2 out_false,
//                    Predicate pred);

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
        const int ia[] = {1, 2, 3, 4, 6, 8, 5, 7};
        int r1[10] = {0};
        int r2[10] = {0};
        typedef std::pair<output_iterator<int*>,  int*> P;
        P p = std::partition_copy(input_iterator<const int*>(std::begin(ia)),
                                  input_iterator<const int*>(std::end(ia)),
                                  output_iterator<int*>(r1), r2, is_odd());
        assert(p.first.base() == r1 + 4);
        assert(r1[0] == 1);
        assert(r1[1] == 3);
        assert(r1[2] == 5);
        assert(r1[3] == 7);
        assert(p.second == r2 + 4);
        assert(r2[0] == 2);
        assert(r2[1] == 4);
        assert(r2[2] == 6);
        assert(r2[3] == 8);
    }
}
