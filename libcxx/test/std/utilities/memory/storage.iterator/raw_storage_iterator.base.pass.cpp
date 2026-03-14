//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_RAW_STORAGE_ITERATOR
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// raw_storage_iterator

#include <memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int A_constructed = 0;

struct A
{
    int data_;
public:
    explicit A(int i) : data_(i) {++A_constructed;}

    A(const A& a) : data_(a.data_)  {++A_constructed;}
    ~A() {--A_constructed; data_ = 0;}

    bool operator==(int i) const {return data_ == i;}
    A* operator& () = delete;
};

int main(int, char**)
{
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

    return 0;
}
