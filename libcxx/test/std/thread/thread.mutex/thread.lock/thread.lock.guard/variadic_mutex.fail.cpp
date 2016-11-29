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

// FIXME: When modules are enabled we can't affect the contents of <mutex>
// by defining a macro
// XFAIL: -fmodules

// <mutex>

// template <class ...Mutex> class lock_guard;

// explicit lock_guard(Mutex&...);

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>

template <class LG>
void test_conversion(LG) {}

int main()
{
    using M = std::mutex;
    M m0, m1, m2;
    M n0, n1, n2;
    {
        using LG = std::lock_guard<>;
        LG lg = {}; // expected-error{{chosen constructor is explicit in copy-initialization}}
        test_conversion<LG>({}); // expected-error{{no matching function for call}}
        ((void)lg);
    }
    {
        using LG = std::lock_guard<M, M>;
        LG lg = {m0, m1}; // expected-error{{chosen constructor is explicit in copy-initialization}}
        test_conversion<LG>({n0, n1}); // expected-error{{no matching function for call}}
        ((void)lg);
    }
    {
        using LG = std::lock_guard<M, M, M>;
        LG lg = {m0, m1, m2}; // expected-error{{chosen constructor is explicit in copy-initialization}}
        test_conversion<LG>({n0, n1, n2}); // expected-error{{no matching function for call}}
    }
}
