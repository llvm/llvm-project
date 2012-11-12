//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Callable<auto, Iter::reference> Function>
//   requires CopyConstructible<Function>
//   Function
//   for_each(Iter first, Iter last, Function f);

#include <algorithm>
#include <cassert>

#include "../../iterators.h"

struct for_each_test
{
    for_each_test(int c) : count(c) {}
    int count;
    void operator()(int& i) {++i; ++count;}
};

int main()
{
    int ia[] = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia)/sizeof(ia[0]);
    for_each_test f = std::for_each(input_iterator<int*>(ia),
                                    input_iterator<int*>(ia+s),
                                    for_each_test(0));
    assert(f.count == s);
    for (unsigned i = 0; i < s; ++i)
        assert(ia[i] == i+1);
}
