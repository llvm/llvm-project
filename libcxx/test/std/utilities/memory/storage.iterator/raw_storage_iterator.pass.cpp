//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14 || c++17

// raw_storage_iterator

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include <MoveOnly.h>

#if TEST_STD_VER >= 11
#define DELETE_FUNCTION = delete
#else
#define DELETE_FUNCTION
#endif

int A_constructed = 0;

struct A
{
    int data_;
public:
    explicit A(int i) : data_(i) {++A_constructed;}

    A(const A& a) : data_(a.data_)  {++A_constructed;}
    ~A() {--A_constructed; data_ = 0;}

    bool operator==(int i) const {return data_ == i;}
    A* operator& () DELETE_FUNCTION;
};

int main(int, char**)
{
    {
    typedef A S;
    typedef std::aligned_storage<3*sizeof(S), std::alignment_of<S>::value>::type
            Storage;
    Storage buffer;
    std::raw_storage_iterator<S*, S> it((S*)&buffer);
    assert(A_constructed == 0);
    for (int i = 0; i < 3; ++i)
    {
        *it++ = S(i+1);
        S* ap = (S*)&buffer + i;
        assert(*ap == i+1);
        assert(A_constructed == i+1);
    }
    }
#if TEST_STD_VER >= 14
    {
    typedef MoveOnly S;
    typedef std::aligned_storage<3*sizeof(S), std::alignment_of<S>::value>::type
            Storage;
    Storage buffer;
    std::raw_storage_iterator<S*, S> it((S*)&buffer);
    S m{1};
    *it++ = std::move(m);
    assert(m.get() == 0); // moved from
    S *ap = (S*) &buffer;
    assert(ap->get() == 1); // original value
    }
#endif

  return 0;
}
