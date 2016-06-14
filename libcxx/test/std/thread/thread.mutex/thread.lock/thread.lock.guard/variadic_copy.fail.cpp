//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <mutex>

// template <class ...Mutex> class lock_guard;

// lock_guard(lock_guard const&) = delete;

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>

int main()
{
    using M = std::mutex;
    M m0, m1, m2;
    {
        using LG = std::lock_guard<>;
        const LG Orig;
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
    {
        using LG = std::lock_guard<M, M>;
        const LG Orig(m0, m1);
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
    {
        using LG = std::lock_guard<M, M, M>;
        const LG Orig(m0, m1, m2);
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
}
