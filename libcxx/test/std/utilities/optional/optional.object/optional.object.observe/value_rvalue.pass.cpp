//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// constexpr T& optional<T>::value() &&;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;
using std::bad_optional_access;

struct X
{
    X() = default;
    X(const X&) = delete;
    constexpr int test() const & {return 3;}
    int test() & {return 4;}
    constexpr int test() const && {return 5;}
    int test() && {return 6;}
};

struct Y
{
    constexpr int test() && {return 7;}
};

constexpr int
test()
{
    optional<Y> opt{Y{}};
    return std::move(opt).value().test();
}

int main(int, char**)
{
    {
        optional<X> opt; ((void)opt);
        ASSERT_NOT_NOEXCEPT(std::move(opt).value());
        ASSERT_SAME_TYPE(decltype(std::move(opt).value()), X&&);
    }
    {
        optional<X> opt;
        opt.emplace();
        assert(std::move(opt).value().test() == 6);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<X> opt;
        try
        {
            (void)std::move(opt).value();
            assert(false);
        }
        catch (const bad_optional_access&)
        {
        }
    }
#endif
    static_assert(test() == 7, "");

  return 0;
}
