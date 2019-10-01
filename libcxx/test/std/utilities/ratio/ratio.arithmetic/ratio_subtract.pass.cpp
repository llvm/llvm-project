//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_subtract

#include <ratio>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::ratio<1, 1> R1;
    typedef std::ratio<1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == 0 && R::den == 1, "");
    }
    {
    typedef std::ratio<1, 2> R1;
    typedef std::ratio<1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 2, "");
    }
    {
    typedef std::ratio<-1, 2> R1;
    typedef std::ratio<1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == -3 && R::den == 2, "");
    }
    {
    typedef std::ratio<1, -2> R1;
    typedef std::ratio<1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == -3 && R::den == 2, "");
    }
    {
    typedef std::ratio<1, 2> R1;
    typedef std::ratio<-1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == 3 && R::den == 2, "");
    }
    {
    typedef std::ratio<1, 2> R1;
    typedef std::ratio<1, -1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == 3 && R::den == 2, "");
    }
    {
    typedef std::ratio<56987354, 467584654> R1;
    typedef std::ratio<544668, 22145> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == -126708206685271LL && R::den == 5177331081415LL, "");
    }
    {
    typedef std::ratio<0> R1;
    typedef std::ratio<0> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == 0 && R::den == 1, "");
    }
    {
    typedef std::ratio<1> R1;
    typedef std::ratio<0> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == 1 && R::den == 1, "");
    }
    {
    typedef std::ratio<0> R1;
    typedef std::ratio<1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
    static_assert(R::num == -1 && R::den == 1, "");
    }

  return 0;
}
