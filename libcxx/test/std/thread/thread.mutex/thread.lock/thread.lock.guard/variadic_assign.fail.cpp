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

// lock_guard& operator=(lock_guard const&) = delete;

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>

int main()
{
    using M = std::mutex;
    M m0, m1, m2;
    M om0, om1, om2;
    {
        using LG = std::lock_guard<>;
        LG lg1, lg2;
        lg1 = lg2; // expected-error{{overload resolution selected deleted operator '='}}
    }
    {
        using LG = std::lock_guard<M, M>;
        LG lg1(m0, m1);
        LG lg2(om0, om1);
        lg1 = lg2; // expected-error{{overload resolution selected deleted operator '='}}
    }
    {
        using LG = std::lock_guard<M, M, M>;
        LG lg1(m0, m1, m2);
        LG lg2(om0, om1, om2);
        lg1 = lg2; // expected-error{{overload resolution selected deleted operator '='}}
    }
}
