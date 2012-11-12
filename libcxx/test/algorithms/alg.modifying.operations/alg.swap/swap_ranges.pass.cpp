//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   Iter2
//   swap_ranges(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
#include <memory>
#endif

#include "../../iterators.h"

template<class Iter1, class Iter2>
void
test()
{
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    Iter2 r = std::swap_ranges(Iter1(i), Iter1(i+3), Iter2(j));
    assert(base(r) == j+3);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

template<class Iter1, class Iter2>
void
test1()
{
    std::unique_ptr<int> i[3];
    for (int k = 0; k < 3; ++k)
        i[k].reset(new int(k+1));
    std::unique_ptr<int> j[3];
    for (int k = 0; k < 3; ++k)
        j[k].reset(new int(k+4));
    Iter2 r = std::swap_ranges(Iter1(i), Iter1(i+3), Iter2(j));
    assert(base(r) == j+3);
    assert(*i[0] == 4);
    assert(*i[1] == 5);
    assert(*i[2] == 6);
    assert(*j[0] == 1);
    assert(*j[1] == 2);
    assert(*j[2] == 3);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    test<forward_iterator<int*>, forward_iterator<int*> >();
    test<forward_iterator<int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<int*>, random_access_iterator<int*> >();
    test<forward_iterator<int*>, int*>();

    test<bidirectional_iterator<int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<int*>, int*>();

    test<random_access_iterator<int*>, forward_iterator<int*> >();
    test<random_access_iterator<int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<int*>, random_access_iterator<int*> >();
    test<random_access_iterator<int*>, int*>();

    test<int*, forward_iterator<int*> >();
    test<int*, bidirectional_iterator<int*> >();
    test<int*, random_access_iterator<int*> >();
    test<int*, int*>();

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    test1<forward_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<bidirectional_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<random_access_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<std::unique_ptr<int>*, forward_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, random_access_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, std::unique_ptr<int>*>();

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
