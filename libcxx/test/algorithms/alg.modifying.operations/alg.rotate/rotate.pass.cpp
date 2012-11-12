//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ShuffleIterator Iter>
//   Iter
//   rotate(Iter first, Iter middle, Iter last);

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
    Iter r = std::rotate(Iter(ia), Iter(ia), Iter(ia));
    assert(base(r) == ia);
    assert(ia[0] == 0);
    r = std::rotate(Iter(ia), Iter(ia), Iter(ia+sa));
    assert(base(r) == ia+sa);
    assert(ia[0] == 0);
    r = std::rotate(Iter(ia), Iter(ia+sa), Iter(ia+sa));
    assert(base(r) == ia);
    assert(ia[0] == 0);

    int ib[] = {0, 1};
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    r = std::rotate(Iter(ib), Iter(ib), Iter(ib+sb));
    assert(base(r) == ib+sb);
    assert(ib[0] == 0);
    assert(ib[1] == 1);
    r = std::rotate(Iter(ib), Iter(ib+1), Iter(ib+sb));
    assert(base(r) == ib+1);
    assert(ib[0] == 1);
    assert(ib[1] == 0);
    r = std::rotate(Iter(ib), Iter(ib+sb), Iter(ib+sb));
    assert(base(r) == ib);
    assert(ib[0] == 1);
    assert(ib[1] == 0);

    int ic[] = {0, 1, 2};
    const unsigned sc = sizeof(ic)/sizeof(ic[0]);
    r = std::rotate(Iter(ic), Iter(ic), Iter(ic+sc));
    assert(base(r) == ic+sc);
    assert(ic[0] == 0);
    assert(ic[1] == 1);
    assert(ic[2] == 2);
    r = std::rotate(Iter(ic), Iter(ic+1), Iter(ic+sc));
    assert(base(r) == ic+2);
    assert(ic[0] == 1);
    assert(ic[1] == 2);
    assert(ic[2] == 0);
    r = std::rotate(Iter(ic), Iter(ic+2), Iter(ic+sc));
    assert(base(r) == ic+1);
    assert(ic[0] == 0);
    assert(ic[1] == 1);
    assert(ic[2] == 2);
    r = std::rotate(Iter(ic), Iter(ic+sc), Iter(ic+sc));
    assert(base(r) == ic);
    assert(ic[0] == 0);
    assert(ic[1] == 1);
    assert(ic[2] == 2);

    int id[] = {0, 1, 2, 3};
    const unsigned sd = sizeof(id)/sizeof(id[0]);
    r = std::rotate(Iter(id), Iter(id), Iter(id+sd));
    assert(base(r) == id+sd);
    assert(id[0] == 0);
    assert(id[1] == 1);
    assert(id[2] == 2);
    assert(id[3] == 3);
    r = std::rotate(Iter(id), Iter(id+1), Iter(id+sd));
    assert(base(r) == id+3);
    assert(id[0] == 1);
    assert(id[1] == 2);
    assert(id[2] == 3);
    assert(id[3] == 0);
    r = std::rotate(Iter(id), Iter(id+2), Iter(id+sd));
    assert(base(r) == id+2);
    assert(id[0] == 3);
    assert(id[1] == 0);
    assert(id[2] == 1);
    assert(id[3] == 2);
    r = std::rotate(Iter(id), Iter(id+3), Iter(id+sd));
    assert(base(r) == id+1);
    assert(id[0] == 2);
    assert(id[1] == 3);
    assert(id[2] == 0);
    assert(id[3] == 1);
    r = std::rotate(Iter(id), Iter(id+sd), Iter(id+sd));
    assert(base(r) == id);
    assert(id[0] == 2);
    assert(id[1] == 3);
    assert(id[2] == 0);
    assert(id[3] == 1);

    int ie[] = {0, 1, 2, 3, 4};
    const unsigned se = sizeof(ie)/sizeof(ie[0]);
    r = std::rotate(Iter(ie), Iter(ie), Iter(ie+se));
    assert(base(r) == ie+se);
    assert(ie[0] == 0);
    assert(ie[1] == 1);
    assert(ie[2] == 2);
    assert(ie[3] == 3);
    assert(ie[4] == 4);
    r = std::rotate(Iter(ie), Iter(ie+1), Iter(ie+se));
    assert(base(r) == ie+4);
    assert(ie[0] == 1);
    assert(ie[1] == 2);
    assert(ie[2] == 3);
    assert(ie[3] == 4);
    assert(ie[4] == 0);
    r = std::rotate(Iter(ie), Iter(ie+2), Iter(ie+se));
    assert(base(r) == ie+3);
    assert(ie[0] == 3);
    assert(ie[1] == 4);
    assert(ie[2] == 0);
    assert(ie[3] == 1);
    assert(ie[4] == 2);
    r = std::rotate(Iter(ie), Iter(ie+3), Iter(ie+se));
    assert(base(r) == ie+2);
    assert(ie[0] == 1);
    assert(ie[1] == 2);
    assert(ie[2] == 3);
    assert(ie[3] == 4);
    assert(ie[4] == 0);
    r = std::rotate(Iter(ie), Iter(ie+4), Iter(ie+se));
    assert(base(r) == ie+1);
    assert(ie[0] == 0);
    assert(ie[1] == 1);
    assert(ie[2] == 2);
    assert(ie[3] == 3);
    assert(ie[4] == 4);
    r = std::rotate(Iter(ie), Iter(ie+se), Iter(ie+se));
    assert(base(r) == ie);
    assert(ie[0] == 0);
    assert(ie[1] == 1);
    assert(ie[2] == 2);
    assert(ie[3] == 3);
    assert(ie[4] == 4);

    int ig[] = {0, 1, 2, 3, 4, 5};
    const unsigned sg = sizeof(ig)/sizeof(ig[0]);
    r = std::rotate(Iter(ig), Iter(ig), Iter(ig+sg));
    assert(base(r) == ig+sg);
    assert(ig[0] == 0);
    assert(ig[1] == 1);
    assert(ig[2] == 2);
    assert(ig[3] == 3);
    assert(ig[4] == 4);
    assert(ig[5] == 5);
    r = std::rotate(Iter(ig), Iter(ig+1), Iter(ig+sg));
    assert(base(r) == ig+5);
    assert(ig[0] == 1);
    assert(ig[1] == 2);
    assert(ig[2] == 3);
    assert(ig[3] == 4);
    assert(ig[4] == 5);
    assert(ig[5] == 0);
    r = std::rotate(Iter(ig), Iter(ig+2), Iter(ig+sg));
    assert(base(r) == ig+4);
    assert(ig[0] == 3);
    assert(ig[1] == 4);
    assert(ig[2] == 5);
    assert(ig[3] == 0);
    assert(ig[4] == 1);
    assert(ig[5] == 2);
    r = std::rotate(Iter(ig), Iter(ig+3), Iter(ig+sg));
    assert(base(r) == ig+3);
    assert(ig[0] == 0);
    assert(ig[1] == 1);
    assert(ig[2] == 2);
    assert(ig[3] == 3);
    assert(ig[4] == 4);
    assert(ig[5] == 5);
    r = std::rotate(Iter(ig), Iter(ig+4), Iter(ig+sg));
    assert(base(r) == ig+2);
    assert(ig[0] == 4);
    assert(ig[1] == 5);
    assert(ig[2] == 0);
    assert(ig[3] == 1);
    assert(ig[4] == 2);
    assert(ig[5] == 3);
    r = std::rotate(Iter(ig), Iter(ig+5), Iter(ig+sg));
    assert(base(r) == ig+1);
    assert(ig[0] == 3);
    assert(ig[1] == 4);
    assert(ig[2] == 5);
    assert(ig[3] == 0);
    assert(ig[4] == 1);
    assert(ig[5] == 2);
    r = std::rotate(Iter(ig), Iter(ig+sg), Iter(ig+sg));
    assert(base(r) == ig);
    assert(ig[0] == 3);
    assert(ig[1] == 4);
    assert(ig[2] == 5);
    assert(ig[3] == 0);
    assert(ig[4] == 1);
    assert(ig[5] == 2);
}

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

template <class Iter>
void
test1()
{
    std::unique_ptr<int> ia[1];
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    for (int i = 0; i < sa; ++i)
        ia[i].reset(new int(i));
    Iter r = std::rotate(Iter(ia), Iter(ia), Iter(ia));
    assert(base(r) == ia);
    assert(*ia[0] == 0);
    r = std::rotate(Iter(ia), Iter(ia), Iter(ia+sa));
    assert(base(r) == ia+sa);
    assert(*ia[0] == 0);
    r = std::rotate(Iter(ia), Iter(ia+sa), Iter(ia+sa));
    assert(base(r) == ia);
    assert(*ia[0] == 0);

    std::unique_ptr<int> ib[2];
    const unsigned sb = sizeof(ib)/sizeof(ib[0]);
    for (int i = 0; i < sb; ++i)
        ib[i].reset(new int(i));
    r = std::rotate(Iter(ib), Iter(ib), Iter(ib+sb));
    assert(base(r) == ib+sb);
    assert(*ib[0] == 0);
    assert(*ib[1] == 1);
    r = std::rotate(Iter(ib), Iter(ib+1), Iter(ib+sb));
    assert(base(r) == ib+1);
    assert(*ib[0] == 1);
    assert(*ib[1] == 0);
    r = std::rotate(Iter(ib), Iter(ib+sb), Iter(ib+sb));
    assert(base(r) == ib);
    assert(*ib[0] == 1);
    assert(*ib[1] == 0);

    std::unique_ptr<int> ic[3];
    const unsigned sc = sizeof(ic)/sizeof(ic[0]);
    for (int i = 0; i < sc; ++i)
        ic[i].reset(new int(i));
    r = std::rotate(Iter(ic), Iter(ic), Iter(ic+sc));
    assert(base(r) == ic+sc);
    assert(*ic[0] == 0);
    assert(*ic[1] == 1);
    assert(*ic[2] == 2);
    r = std::rotate(Iter(ic), Iter(ic+1), Iter(ic+sc));
    assert(base(r) == ic+2);
    assert(*ic[0] == 1);
    assert(*ic[1] == 2);
    assert(*ic[2] == 0);
    r = std::rotate(Iter(ic), Iter(ic+2), Iter(ic+sc));
    assert(base(r) == ic+1);
    assert(*ic[0] == 0);
    assert(*ic[1] == 1);
    assert(*ic[2] == 2);
    r = std::rotate(Iter(ic), Iter(ic+sc), Iter(ic+sc));
    assert(base(r) == ic);
    assert(*ic[0] == 0);
    assert(*ic[1] == 1);
    assert(*ic[2] == 2);

    std::unique_ptr<int> id[4];
    const unsigned sd = sizeof(id)/sizeof(id[0]);
    for (int i = 0; i < sd; ++i)
        id[i].reset(new int(i));
    r = std::rotate(Iter(id), Iter(id), Iter(id+sd));
    assert(base(r) == id+sd);
    assert(*id[0] == 0);
    assert(*id[1] == 1);
    assert(*id[2] == 2);
    assert(*id[3] == 3);
    r = std::rotate(Iter(id), Iter(id+1), Iter(id+sd));
    assert(base(r) == id+3);
    assert(*id[0] == 1);
    assert(*id[1] == 2);
    assert(*id[2] == 3);
    assert(*id[3] == 0);
    r = std::rotate(Iter(id), Iter(id+2), Iter(id+sd));
    assert(base(r) == id+2);
    assert(*id[0] == 3);
    assert(*id[1] == 0);
    assert(*id[2] == 1);
    assert(*id[3] == 2);
    r = std::rotate(Iter(id), Iter(id+3), Iter(id+sd));
    assert(base(r) == id+1);
    assert(*id[0] == 2);
    assert(*id[1] == 3);
    assert(*id[2] == 0);
    assert(*id[3] == 1);
    r = std::rotate(Iter(id), Iter(id+sd), Iter(id+sd));
    assert(base(r) == id);
    assert(*id[0] == 2);
    assert(*id[1] == 3);
    assert(*id[2] == 0);
    assert(*id[3] == 1);

    std::unique_ptr<int> ie[5];
    const unsigned se = sizeof(ie)/sizeof(ie[0]);
    for (int i = 0; i < se; ++i)
        ie[i].reset(new int(i));
    r = std::rotate(Iter(ie), Iter(ie), Iter(ie+se));
    assert(base(r) == ie+se);
    assert(*ie[0] == 0);
    assert(*ie[1] == 1);
    assert(*ie[2] == 2);
    assert(*ie[3] == 3);
    assert(*ie[4] == 4);
    r = std::rotate(Iter(ie), Iter(ie+1), Iter(ie+se));
    assert(base(r) == ie+4);
    assert(*ie[0] == 1);
    assert(*ie[1] == 2);
    assert(*ie[2] == 3);
    assert(*ie[3] == 4);
    assert(*ie[4] == 0);
    r = std::rotate(Iter(ie), Iter(ie+2), Iter(ie+se));
    assert(base(r) == ie+3);
    assert(*ie[0] == 3);
    assert(*ie[1] == 4);
    assert(*ie[2] == 0);
    assert(*ie[3] == 1);
    assert(*ie[4] == 2);
    r = std::rotate(Iter(ie), Iter(ie+3), Iter(ie+se));
    assert(base(r) == ie+2);
    assert(*ie[0] == 1);
    assert(*ie[1] == 2);
    assert(*ie[2] == 3);
    assert(*ie[3] == 4);
    assert(*ie[4] == 0);
    r = std::rotate(Iter(ie), Iter(ie+4), Iter(ie+se));
    assert(base(r) == ie+1);
    assert(*ie[0] == 0);
    assert(*ie[1] == 1);
    assert(*ie[2] == 2);
    assert(*ie[3] == 3);
    assert(*ie[4] == 4);
    r = std::rotate(Iter(ie), Iter(ie+se), Iter(ie+se));
    assert(base(r) == ie);
    assert(*ie[0] == 0);
    assert(*ie[1] == 1);
    assert(*ie[2] == 2);
    assert(*ie[3] == 3);
    assert(*ie[4] == 4);

    std::unique_ptr<int> ig[6];
    const unsigned sg = sizeof(ig)/sizeof(ig[0]);
    for (int i = 0; i < sg; ++i)
        ig[i].reset(new int(i));
    r = std::rotate(Iter(ig), Iter(ig), Iter(ig+sg));
    assert(base(r) == ig+sg);
    assert(*ig[0] == 0);
    assert(*ig[1] == 1);
    assert(*ig[2] == 2);
    assert(*ig[3] == 3);
    assert(*ig[4] == 4);
    assert(*ig[5] == 5);
    r = std::rotate(Iter(ig), Iter(ig+1), Iter(ig+sg));
    assert(base(r) == ig+5);
    assert(*ig[0] == 1);
    assert(*ig[1] == 2);
    assert(*ig[2] == 3);
    assert(*ig[3] == 4);
    assert(*ig[4] == 5);
    assert(*ig[5] == 0);
    r = std::rotate(Iter(ig), Iter(ig+2), Iter(ig+sg));
    assert(base(r) == ig+4);
    assert(*ig[0] == 3);
    assert(*ig[1] == 4);
    assert(*ig[2] == 5);
    assert(*ig[3] == 0);
    assert(*ig[4] == 1);
    assert(*ig[5] == 2);
    r = std::rotate(Iter(ig), Iter(ig+3), Iter(ig+sg));
    assert(base(r) == ig+3);
    assert(*ig[0] == 0);
    assert(*ig[1] == 1);
    assert(*ig[2] == 2);
    assert(*ig[3] == 3);
    assert(*ig[4] == 4);
    assert(*ig[5] == 5);
    r = std::rotate(Iter(ig), Iter(ig+4), Iter(ig+sg));
    assert(base(r) == ig+2);
    assert(*ig[0] == 4);
    assert(*ig[1] == 5);
    assert(*ig[2] == 0);
    assert(*ig[3] == 1);
    assert(*ig[4] == 2);
    assert(*ig[5] == 3);
    r = std::rotate(Iter(ig), Iter(ig+5), Iter(ig+sg));
    assert(base(r) == ig+1);
    assert(*ig[0] == 3);
    assert(*ig[1] == 4);
    assert(*ig[2] == 5);
    assert(*ig[3] == 0);
    assert(*ig[4] == 1);
    assert(*ig[5] == 2);
    r = std::rotate(Iter(ig), Iter(ig+sg), Iter(ig+sg));
    assert(base(r) == ig);
    assert(*ig[0] == 3);
    assert(*ig[1] == 4);
    assert(*ig[2] == 5);
    assert(*ig[3] == 0);
    assert(*ig[4] == 1);
    assert(*ig[5] == 2);
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

    test1<forward_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*>();

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
