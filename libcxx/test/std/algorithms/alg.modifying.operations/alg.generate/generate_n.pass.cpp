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

#include "test_macros.h"
#include "test_iterators.h"
#include "user_defined_integral.hpp"

struct gen_test
{
    int operator()() const {return 2;}
};

template <class Iter, class Size>
void
test2()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::generate_n(Iter(ia), Size(n), gen_test()) == Iter(ia+n));
    assert(ia[0] == 2);
    assert(ia[1] == 2);
    assert(ia[2] == 2);
    assert(ia[3] == 2);
}

template <class Iter>
void
test()
{
    test2<Iter, int>();
    test2<Iter, unsigned int>();
    test2<Iter, long>();
    test2<Iter, unsigned long>();
    test2<Iter, UserDefinedIntegral<unsigned> >();
    test2<Iter, float>();
    test2<Iter, double>();  // this is PR#35498
    test2<Iter, long double>();
}

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();
}
