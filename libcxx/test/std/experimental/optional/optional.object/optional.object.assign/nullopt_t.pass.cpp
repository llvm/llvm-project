//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// optional<T>& operator=(nullopt_t) noexcept;

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;
using std::experimental::nullopt_t;
using std::experimental::nullopt;

struct X
{
    static bool dtor_called;
    ~X() {dtor_called = true;}
};

bool X::dtor_called = false;

int main()
{
    {
        optional<int> opt;
        static_assert(noexcept(opt = nullopt) == true, "");
        opt = nullopt;
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<int> opt(3);
        opt = nullopt;
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<X> opt;
        static_assert(noexcept(opt = nullopt) == true, "");
        assert(X::dtor_called == false);
        opt = nullopt;
        assert(X::dtor_called == false);
        assert(static_cast<bool>(opt) == false);
    }
    {
        X x;
        {
            optional<X> opt(x);
            assert(X::dtor_called == false);
            opt = nullopt;
            assert(X::dtor_called == true);
            assert(static_cast<bool>(opt) == false);
        }
    }
}
