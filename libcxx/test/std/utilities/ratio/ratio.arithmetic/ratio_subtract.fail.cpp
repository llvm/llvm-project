//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_subtract

#include <ratio>

int main()
{
    typedef std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef std::ratio<1, 1> R2;
    typedef std::ratio_subtract<R1, R2>::type R;
}
