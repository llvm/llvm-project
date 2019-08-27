//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// optional<T>& operator=(nullopt_t) noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;
using std::nullopt_t;
using std::nullopt;

int main(int, char**)
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
    using TT = TestTypes::TestType;
    TT::reset();
    {
        optional<TT> opt;
        static_assert(noexcept(opt = nullopt) == true, "");
        assert(TT::destroyed == 0);
        opt = nullopt;
        assert(TT::constructed == 0);
        assert(TT::alive == 0);
        assert(TT::destroyed == 0);
        assert(static_cast<bool>(opt) == false);
    }
    assert(TT::alive == 0);
    assert(TT::destroyed == 0);
    TT::reset();
    {
        optional<TT> opt(42);
        assert(TT::destroyed == 0);
        TT::reset_constructors();
        opt = nullopt;
        assert(TT::constructed == 0);
        assert(TT::alive == 0);
        assert(TT::destroyed == 1);
        assert(static_cast<bool>(opt) == false);
    }
    assert(TT::alive == 0);
    assert(TT::destroyed == 1);
    TT::reset();

  return 0;
}
