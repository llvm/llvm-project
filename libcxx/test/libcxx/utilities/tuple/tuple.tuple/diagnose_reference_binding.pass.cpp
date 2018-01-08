//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <tuple>

// Test the diagnostics libc++ generates for invalid reference binding.
// Libc++ attempts to diagnose the following cases:
//  * Constructing an lvalue reference from an rvalue.
//  * Constructing an rvalue reference from an lvalue.

#include <tuple>
#include <string>
#include <functional>
#include <cassert>

static_assert(std::is_constructible<int&, std::reference_wrapper<int>>::value, "");
static_assert(std::is_constructible<int const&, std::reference_wrapper<int>>::value, "");


int main() {
    std::allocator<void> alloc;
    int x = 42;
    {
        std::tuple<int&> t(std::ref(x));
        assert(&std::get<0>(t) == &x);
        std::tuple<int&> t1(std::allocator_arg, alloc, std::ref(x));
        assert(&std::get<0>(t1) == &x);
    }
    {
        auto r = std::ref(x);
        auto const& cr = r;
        std::tuple<int&> t(r);
        assert(&std::get<0>(t) == &x);
        std::tuple<int&> t1(cr);
        assert(&std::get<0>(t1) == &x);
        std::tuple<int&> t2(std::allocator_arg, alloc, r);
        assert(&std::get<0>(t2) == &x);
        std::tuple<int&> t3(std::allocator_arg, alloc, cr);
        assert(&std::get<0>(t3) == &x);
    }
    {
        std::tuple<int const&> t(std::ref(x));
        assert(&std::get<0>(t) == &x);
        std::tuple<int const&> t2(std::cref(x));
        assert(&std::get<0>(t2) == &x);
        std::tuple<int const&> t3(std::allocator_arg, alloc, std::ref(x));
        assert(&std::get<0>(t3) == &x);
        std::tuple<int const&> t4(std::allocator_arg, alloc, std::cref(x));
        assert(&std::get<0>(t4) == &x);
    }
    {
        auto r = std::ref(x);
        auto cr = std::cref(x);
        std::tuple<int const&> t(r);
        assert(&std::get<0>(t) == &x);
        std::tuple<int const&> t2(cr);
        assert(&std::get<0>(t2) == &x);
        std::tuple<int const&> t3(std::allocator_arg, alloc, r);
        assert(&std::get<0>(t3) == &x);
        std::tuple<int const&> t4(std::allocator_arg, alloc, cr);
        assert(&std::get<0>(t4) == &x);
    }
}
