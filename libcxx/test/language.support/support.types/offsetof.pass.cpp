//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#ifndef offsetof
#error offsetof not defined
#endif

struct A
{
    int x;
};

int main()
{
#if (__has_feature(cxx_noexcept))
    static_assert(noexcept(offsetof(A, x)), "");
#endif
}
