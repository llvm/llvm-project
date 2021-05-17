//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9 && !no-exceptions

// <optional>

// constexpr const T& optional<T>::value() const &&;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;
using std::in_place_t;
using std::in_place;
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

int main(int, char**)
{
    {
        const optional<X> opt; ((void)opt);
        ASSERT_NOT_NOEXCEPT(std::move(opt).value());
        ASSERT_SAME_TYPE(decltype(std::move(opt).value()), X const&&);
    }
    {
        constexpr optional<X> opt(in_place);
        static_assert(std::move(opt).value().test() == 5, "");
    }
    {
        const optional<X> opt(in_place);
        assert(std::move(opt).value().test() == 5);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        const optional<X> opt;
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

  return 0;
}
