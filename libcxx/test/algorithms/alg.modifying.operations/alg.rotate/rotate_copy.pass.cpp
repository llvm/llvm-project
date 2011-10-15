//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   OutIter
//   rotate_copy(InIter first, InIter middle, InIter last, OutIter result);

#include <algorithm>
#include <cassert>

#include "../../iterators.h"

template <class InIter, class OutIter>
void
test()
{
    int ia[] = {0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[sa] = {0};

    OutIter r = std::rotate_copy(InIter(ia), InIter(ia), InIter(ia), OutIter(ib));
    assert(base(r) == ib);

    r = std::rotate_copy(InIter(ia), InIter(ia), InIter(ia+1), OutIter(ib));
    assert(base(r) == ib+1);
    assert(ib[0] == 0);

    r = std::rotate_copy(InIter(ia), InIter(ia+1), InIter(ia+1), OutIter(ib));
    assert(base(r) == ib+1);
    assert(ib[0] == 0);

    r = std::rotate_copy(InIter(ia), InIter(ia), InIter(ia+2), OutIter(ib));
    assert(base(r) == ib+2);
    assert(ib[0] == 0);
    assert(ib[1] == 1);

    r = std::rotate_copy(InIter(ia), InIter(ia+1), InIter(ia+2), OutIter(ib));
    assert(base(r) == ib+2);
    assert(ib[0] == 1);
    assert(ib[1] == 0);

    r = std::rotate_copy(InIter(ia), InIter(ia+2), InIter(ia+2), OutIter(ib));
    assert(base(r) == ib+2);
    assert(ib[0] == 0);
    assert(ib[1] == 1);

    r = std::rotate_copy(InIter(ia), InIter(ia), InIter(ia+3), OutIter(ib));
    assert(base(r) == ib+3);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    assert(ib[2] == 2);

    r = std::rotate_copy(InIter(ia), InIter(ia+1), InIter(ia+3), OutIter(ib));
    assert(base(r) == ib+3);
    assert(ib[0] == 1);
    assert(ib[1] == 2);
    assert(ib[2] == 0);

    r = std::rotate_copy(InIter(ia), InIter(ia+2), InIter(ia+3), OutIter(ib));
    assert(base(r) == ib+3);
    assert(ib[0] == 2);
    assert(ib[1] == 0);
    assert(ib[2] == 1);

    r = std::rotate_copy(InIter(ia), InIter(ia+3), InIter(ia+3), OutIter(ib));
    assert(base(r) == ib+3);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    assert(ib[2] == 2);

    r = std::rotate_copy(InIter(ia), InIter(ia), InIter(ia+4), OutIter(ib));
    assert(base(r) == ib+4);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    assert(ib[2] == 2);
    assert(ib[3] == 3);

    r = std::rotate_copy(InIter(ia), InIter(ia+1), InIter(ia+4), OutIter(ib));
    assert(base(r) == ib+4);
    assert(ib[0] == 1);
    assert(ib[1] == 2);
    assert(ib[2] == 3);
    assert(ib[3] == 0);

    r = std::rotate_copy(InIter(ia), InIter(ia+2), InIter(ia+4), OutIter(ib));
    assert(base(r) == ib+4);
    assert(ib[0] == 2);
    assert(ib[1] == 3);
    assert(ib[2] == 0);
    assert(ib[3] == 1);

    r = std::rotate_copy(InIter(ia), InIter(ia+3), InIter(ia+4), OutIter(ib));
    assert(base(r) == ib+4);
    assert(ib[0] == 3);
    assert(ib[1] == 0);
    assert(ib[2] == 1);
    assert(ib[3] == 2);

    r = std::rotate_copy(InIter(ia), InIter(ia+4), InIter(ia+4), OutIter(ib));
    assert(base(r) == ib+4);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    assert(ib[2] == 2);
    assert(ib[3] == 3);
}

int main()
{
    test<bidirectional_iterator<const int*>, output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, output_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, int*>();

    test<const int*, output_iterator<int*> >();
    test<const int*, forward_iterator<int*> >();
    test<const int*, bidirectional_iterator<int*> >();
    test<const int*, random_access_iterator<int*> >();
    test<const int*, int*>();
}
