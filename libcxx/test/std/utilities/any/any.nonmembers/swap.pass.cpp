//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// XFAIL: availability=macosx10.13
// XFAIL: availability=macosx10.12
// XFAIL: availability=macosx10.11
// XFAIL: availability=macosx10.10
// XFAIL: availability=macosx10.9
// XFAIL: availability=macosx10.8
// XFAIL: availability=macosx10.7

// <any>

// void swap(any &, any &) noexcept

// swap(...) just wraps any::swap(...). That function is tested elsewhere.

#include <any>
#include <cassert>

using std::any;
using std::any_cast;

int main(int, char**)
{

    { // test noexcept
        any a;
        static_assert(noexcept(swap(a, a)), "swap(any&, any&) must be noexcept");
    }
    {
        any a1(1);
        any a2(2);

        swap(a1, a2);

        assert(any_cast<int>(a1) == 2);
        assert(any_cast<int>(a2) == 1);
    }

  return 0;
}
