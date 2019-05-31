//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <optional>

// void reset() noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

struct X
{
    static bool dtor_called;
    ~X() {dtor_called = true;}
};

bool X::dtor_called = false;

int main(int, char**)
{
    {
        optional<int> opt;
        static_assert(noexcept(opt.reset()) == true, "");
        opt.reset();
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<int> opt(3);
        opt.reset();
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<X> opt;
        static_assert(noexcept(opt.reset()) == true, "");
        assert(X::dtor_called == false);
        opt.reset();
        assert(X::dtor_called == false);
        assert(static_cast<bool>(opt) == false);
    }
    {
        optional<X> opt(X{});
        X::dtor_called = false;
        opt.reset();
        assert(X::dtor_called == true);
        assert(static_cast<bool>(opt) == false);
        X::dtor_called = false;
    }

  return 0;
}
