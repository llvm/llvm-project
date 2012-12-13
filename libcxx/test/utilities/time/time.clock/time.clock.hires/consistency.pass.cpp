//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// high_resolution_clock

// check clock invariants

#include <chrono>

template <class _Tp>
void test(const _Tp &) {}

int main()
{
    typedef std::chrono::high_resolution_clock C;
    static_assert((std::is_same<C::rep, C::duration::rep>::value), "");
    static_assert((std::is_same<C::period, C::duration::period>::value), "");
    static_assert((std::is_same<C::duration, C::time_point::duration>::value), "");
    static_assert(C::is_steady || !C::is_steady, "");
    test(std::chrono::high_resolution_clock::is_steady);
}
