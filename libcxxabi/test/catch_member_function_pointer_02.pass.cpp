//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can a noexcept member function pointer be caught by a non-noexcept catch clause?
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions

// Support for catching a function pointer including noexcept was shipped in macOS 10.13
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// GCC supports noexcept function types but this test still fails.
// This is likely a bug in their implementation. Investigation needed.
// XFAIL: gcc-11, gcc-12

#include <cassert>

struct X {
  template<bool Noexcept> void f() noexcept(Noexcept) {}
};
template<bool Noexcept> using FnType = void (X::*)() noexcept(Noexcept);

template<bool ThrowNoexcept, bool CatchNoexcept>
void check()
{
    try
    {
        auto p = &X::f<ThrowNoexcept>;
        throw p;
        assert(false);
    }
    catch (FnType<CatchNoexcept> p)
    {
        assert(ThrowNoexcept || !CatchNoexcept);
        assert(p == &X::f<ThrowNoexcept>);
    }
    catch (...)
    {
        assert(!ThrowNoexcept && CatchNoexcept);
    }
}

void check_deep() {
    FnType<true> p = &X::f<true>;
    try
    {
        throw &p;
    }
    catch (FnType<false> *q)
    {
        assert(false);
    }
    catch (FnType<true> *q)
    {
    }
    catch (...)
    {
        assert(false);
    }
}

int main(int, char**)
{
    check<false, false>();
    check<false, true>();
    check<true, false>();
    check<true, true>();
    check_deep();

    return 0;
}
