//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires OutputIterator<Iter, Iter::reference>
//         && EqualityComparable<Iter::value_type>
//   Iter
//   unique(Iter first, Iter last);

#include <algorithm>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
#include <memory>
#endif

#include "../../iterators.h"

template <class Iter>
void
test()
{
    int ia[] = {0};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    Iter r = std::unique(Iter(ia), Iter(ia+sa));
    assert(base(r) == ia + sa);
    assert(ia[0] == 0);

    int ib[] = {0, 1};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    r = std::unique(Iter(ib), Iter(ib+sb));
    assert(base(r) == ib + sb);
    assert(ib[0] == 0);
    assert(ib[1] == 1);

    int ic[] = {0, 0};
    const unsigned sc = sizeof(ic)/sizeof(ic[0]);
    r = std::unique(Iter(ic), Iter(ic+sc));
    assert(base(r) == ic + 1);
    assert(ic[0] == 0);

    int id[] = {0, 0, 1};
    const unsigned sd = sizeof(id)/sizeof(id[0]);
    r = std::unique(Iter(id), Iter(id+sd));
    assert(base(r) == id + 2);
    assert(id[0] == 0);
    assert(id[1] == 1);

    int ie[] = {0, 0, 1, 0};
    const unsigned se = sizeof(ie)/sizeof(ie[0]);
    r = std::unique(Iter(ie), Iter(ie+se));
    assert(base(r) == ie + 3);
    assert(ie[0] == 0);
    assert(ie[1] == 1);
    assert(ie[2] == 0);

    int ig[] = {0, 0, 1, 1};
    const unsigned sg = sizeof(ig)/sizeof(ig[0]);
    r = std::unique(Iter(ig), Iter(ig+sg));
    assert(base(r) == ig + 2);
    assert(ig[0] == 0);
    assert(ig[1] == 1);

    int ih[] = {0, 1, 1};
    const unsigned sh = sizeof(ih)/sizeof(ih[0]);
    r = std::unique(Iter(ih), Iter(ih+sh));
    assert(base(r) == ih + 2);
    assert(ih[0] == 0);
    assert(ih[1] == 1);

    int ii[] = {0, 1, 1, 1, 2, 2, 2};
    const unsigned si = sizeof(ii)/sizeof(ii[0]);
    r = std::unique(Iter(ii), Iter(ii+si));
    assert(base(r) == ii + 3);
    assert(ii[0] == 0);
    assert(ii[1] == 1);
    assert(ii[2] == 2);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

struct do_nothing
{
    void operator()(void*) const {}
};

typedef std::unique_ptr<int, do_nothing> Ptr;

template <class Iter>
void
test1()
{
    int one = 1;
    int two = 2;
    Ptr ia[1];
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    Iter r = std::unique(Iter(ia), Iter(ia+sa));
    assert(base(r) == ia + sa);
    assert(ia[0] == 0);

    Ptr ib[2];
    ib[1].reset(&one);
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    r = std::unique(Iter(ib), Iter(ib+sb));
    assert(base(r) == ib + sb);
    assert(ib[0] == 0);
    assert(*ib[1] == 1);

    Ptr ic[2];
    const unsigned sc = sizeof(ic)/sizeof(ic[0]);
    r = std::unique(Iter(ic), Iter(ic+sc));
    assert(base(r) == ic + 1);
    assert(ic[0] == 0);

    Ptr id[3];
    id[2].reset(&one);
    const unsigned sd = sizeof(id)/sizeof(id[0]);
    r = std::unique(Iter(id), Iter(id+sd));
    assert(base(r) == id + 2);
    assert(id[0] == 0);
    assert(*id[1] == 1);

    Ptr ie[4];
    ie[2].reset(&one);
    const unsigned se = sizeof(ie)/sizeof(ie[0]);
    r = std::unique(Iter(ie), Iter(ie+se));
    assert(base(r) == ie + 3);
    assert(ie[0] == 0);
    assert(*ie[1] == 1);
    assert(ie[2] == 0);

    Ptr ig[4];
    ig[2].reset(&one);
    ig[3].reset(&one);
    const unsigned sg = sizeof(ig)/sizeof(ig[0]);
    r = std::unique(Iter(ig), Iter(ig+sg));
    assert(base(r) == ig + 2);
    assert(ig[0] == 0);
    assert(*ig[1] == 1);

    Ptr ih[3];
    ih[1].reset(&one);
    ih[2].reset(&one);
    const unsigned sh = sizeof(ih)/sizeof(ih[0]);
    r = std::unique(Iter(ih), Iter(ih+sh));
    assert(base(r) == ih + 2);
    assert(ih[0] == 0);
    assert(*ih[1] == 1);

    Ptr ii[7];
    ii[1].reset(&one);
    ii[2].reset(&one);
    ii[3].reset(&one);
    ii[4].reset(&two);
    ii[5].reset(&two);
    ii[6].reset(&two);
    const unsigned si = sizeof(ii)/sizeof(ii[0]);
    r = std::unique(Iter(ii), Iter(ii+si));
    assert(base(r) == ii + 3);
    assert(ii[0] == 0);
    assert(*ii[1] == 1);
    assert(*ii[2] == 2);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    test1<forward_iterator<Ptr*> >();
    test1<bidirectional_iterator<Ptr*> >();
    test1<random_access_iterator<Ptr*> >();
    test1<Ptr*>();

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
