//===--------------- catch_member_function_pointer_02.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can a noexcept member function pointer be caught by a non-noexcept catch
// clause?
// UNSUPPORTED: no-exceptions, libcxxabi-no-noexcept-function-type

// GCC 7 and 8 support noexcept function types but this test still fails.
// This is likely a bug in their implementation. Investigation needed.
// XFAIL: gcc-7, gcc-8, gcc-9, gcc-10

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

int main()
{
    check<false, false>();
    check<false, true>();
    check<true, false>();
    check<true, true>();
    check_deep();
}
