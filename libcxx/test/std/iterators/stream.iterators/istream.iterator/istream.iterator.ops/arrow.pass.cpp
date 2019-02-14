//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// const T* operator->() const;

#include <iterator>
#include <sstream>
#include <cassert>

struct A
{
    double d_;
    int i_;
};

void operator&(A const&) {}

std::istream& operator>>(std::istream& is, A& a)
{
    return is >> a.d_ >> a.i_;
}

int main(int, char**)
{
    std::istringstream inf("1.5  23 ");
    std::istream_iterator<A> i(inf);
    assert(i->d_ == 1.5);
    assert(i->i_ == 23);

  return 0;
}
