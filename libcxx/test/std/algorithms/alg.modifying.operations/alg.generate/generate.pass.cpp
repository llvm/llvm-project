//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   constexpr void      // constexpr after c++17
//   generate(Iter first, Iter last, Generator gen);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct gen_test
{
    TEST_CONSTEXPR int operator()() const {return 1;}
};


#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {0, 1, 2, 3, 4};

    std::generate(std::begin(ia), std::end(ia), gen_test());

    return std::all_of(std::begin(ia), std::end(ia), [](int x) { return x == 1; })
        ;
    }
#endif


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

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif
}
