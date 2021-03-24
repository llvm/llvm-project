//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: LIBCXX-WINDOWS-FIXME

// <functional>

// default searcher
// template<class _ForwardIterator, class _BinaryPredicate = equal_to<>>
// class default_searcher {
// public:
//     default_searcher(_ForwardIterator __f, _ForwardIterator __l,
//                             _BinaryPredicate __p = _BinaryPredicate())
//         : __first_(__f), __last_(__l), __pred_(__p) {}
//
//     template <typename _ForwardIterator2>
//     pair<_ForwardIterator2, _ForwardIterator2>
//     operator () (_ForwardIterator2 __f, _ForwardIterator2 __l) const {
//         return std::search(__f, __l, __first_, __last_, __pred_);
//         }
//
// private:
//     _ForwardIterator __first_;
//     _ForwardIterator __last_;
//     _BinaryPredicate __pred_;
//     };


#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <typename Iter1, typename Iter2>
TEST_CONSTEXPR_CXX20
void do_search(Iter1 b1, Iter1 e1, Iter2 b2, Iter2 e2, Iter1 result) {
    std::default_searcher<Iter2> s{b2, e2};
    assert(result == std::search(b1, e1, s));
}

template <class Iter1, class Iter2>
TEST_CONSTEXPR_CXX20
bool test()
{
    int ia[] = {0, 1, 2, 3, 4, 5};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia),      Iter2(ia),    Iter1(ia));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia),      Iter2(ia+1),  Iter1(ia));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+1),    Iter2(ia+2),  Iter1(ia+1));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+2),    Iter2(ia+2),  Iter1(ia));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+2),    Iter2(ia+3),  Iter1(ia+2));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+2),    Iter2(ia+3),  Iter1(ia+2));
    do_search(Iter1(ia), Iter1(ia),      Iter2(ia+2),    Iter2(ia+3),  Iter1(ia));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+sa-1), Iter2(ia+sa), Iter1(ia+sa-1));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia+sa-3), Iter2(ia+sa), Iter1(ia+sa-3));
    do_search(Iter1(ia), Iter1(ia+sa),   Iter2(ia),      Iter2(ia+sa), Iter1(ia));
    do_search(Iter1(ia), Iter1(ia+sa-1), Iter2(ia),      Iter2(ia+sa), Iter1(ia+sa-1));
    do_search(Iter1(ia), Iter1(ia+1),    Iter2(ia),      Iter2(ia+sa), Iter1(ia+1));
    int ib[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    int ic[] = {1};
    do_search(Iter1(ib), Iter1(ib+sb), Iter2(ic), Iter2(ic+1), Iter1(ib+1));
    int id[] = {1, 2};
    do_search(Iter1(ib), Iter1(ib+sb), Iter2(id), Iter2(id+2), Iter1(ib+1));
    int ie[] = {1, 2, 3};
    do_search(Iter1(ib), Iter1(ib+sb), Iter2(ie), Iter2(ie+3), Iter1(ib+4));
    int ig[] = {1, 2, 3, 4};
    do_search(Iter1(ib), Iter1(ib+sb), Iter2(ig), Iter2(ig+4), Iter1(ib+8));
    int ih[] = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
    const unsigned sh = sizeof(ih)/sizeof(ih[0]);
    int ii[] = {1, 1, 2};
    do_search(Iter1(ih), Iter1(ih+sh), Iter2(ii), Iter2(ii+3), Iter1(ih+3));
    int ij[] = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
    const unsigned sj = sizeof(ij)/sizeof(ij[0]);
    int ik[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
    const unsigned sk = sizeof(ik)/sizeof(ik[0]);
    do_search(Iter1(ij), Iter1(ij+sj), Iter2(ik), Iter2(ik+sk), Iter1(ij+6));

    return true;
}

int main(int, char**) {
    test<forward_iterator<const int*>, forward_iterator<const int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*> >();

#if TEST_STD_VER >= 20
    static_assert(test<forward_iterator<const int*>, forward_iterator<const int*>>());
    static_assert(test<forward_iterator<const int*>, bidirectional_iterator<const int*>>());
    static_assert(test<forward_iterator<const int*>, random_access_iterator<const int*>>());
    static_assert(test<bidirectional_iterator<const int*>, forward_iterator<const int*>>());
    static_assert(test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>());
    static_assert(test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>());
    static_assert(test<random_access_iterator<const int*>, forward_iterator<const int*>>());
    static_assert(test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>());
    static_assert(test<random_access_iterator<const int*>, random_access_iterator<const int*>>());
#endif

    return 0;
}
