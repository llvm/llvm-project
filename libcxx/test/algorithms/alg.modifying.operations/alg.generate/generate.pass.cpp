//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   void
//   generate(Iter first, Iter last, Generator gen);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

struct gen_test
{
    int operator()() const {return 1;}
};

template <class Iter>
void
test()
{
    const unsigned n = 4;
    int ia[n] = {0};
    std::generate(Iter(ia), Iter(ia+n), gen_test());
    assert(ia[0] == 1);
    assert(ia[1] == 1);
    assert(ia[2] == 1);
    assert(ia[3] == 1);
}

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();
}
