//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   void
//   generate_n(Iter first, Size n, Generator gen);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

struct gen_test
{
    int operator()() const {return 2;}
};

template <class Iter>
void
test()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::generate_n(Iter(ia), n, gen_test()) == Iter(ia+n));
    assert(ia[0] == 2);
    assert(ia[1] == 2);
    assert(ia[2] == 2);
    assert(ia[3] == 2);
}

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();
}
