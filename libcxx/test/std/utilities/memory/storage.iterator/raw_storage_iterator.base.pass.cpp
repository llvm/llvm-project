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
#if TEST_STD_VER >= 14
    typedef std::aligned_storage<3*sizeof(A), std::alignment_of<A>::value>::type
            Storage;
    Storage buffer;
    std::raw_storage_iterator<A*, A> it((A*)&buffer);
    assert(A_constructed == 0);
    assert(it.base() == (A*)&buffer);
    for (int i = 0; i < 3; ++i)
    {
        *it++ = A(i+1);
        A* ap = (A*)&buffer + i;
        assert(*ap == i+1);
        assert(A_constructed == i+1);
        assert(it.base() == ap + 1);  // next place to write
    }
#endif

  return 0;
}
