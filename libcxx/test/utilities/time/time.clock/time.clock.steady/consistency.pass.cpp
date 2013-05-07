//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This test uses new symbols that were not defined in the libc++ shipped on
// darwin11 and darwin12:
// XFAIL: with_system_lib=x86_64-apple-darwin11
// XFAIL: with_system_lib=x86_64-apple-darwin12

// <chrono>

// steady_clock

// check clock invariants

#include <chrono>

template <class _Tp>
void test(const _Tp &) {}

int main()
{
    typedef std::chrono::steady_clock C;
    static_assert((std::is_same<C::rep, C::duration::rep>::value), "");
    static_assert((std::is_same<C::period, C::duration::period>::value), "");
    static_assert((std::is_same<C::duration, C::time_point::duration>::value), "");
    static_assert(C::is_steady, "");
    test(std::chrono::steady_clock::is_steady);
}
