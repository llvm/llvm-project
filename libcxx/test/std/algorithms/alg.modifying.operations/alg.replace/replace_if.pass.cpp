//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, Predicate<auto, Iter::value_type> Pred, class T>
//   requires OutputIterator<Iter, Iter::reference>
//         && OutputIterator<Iter, const T&>
//         && CopyConstructible<Pred>
//   void
//   replace_if(Iter first, Iter last, Pred pred, const T& new_value);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_iterators.h"

bool equalToTwo(int v) { return v == 2; }

template <class Iter>
void
test()
{
    int ia[] = {0, 1, 2, 3, 4};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    std::replace_if(Iter(ia), Iter(ia+sa), equalToTwo, 5);
    assert(ia[0] == 0);
    assert(ia[1] == 1);
    assert(ia[2] == 5);
    assert(ia[3] == 3);
    assert(ia[4] == 4);
}

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();
}
