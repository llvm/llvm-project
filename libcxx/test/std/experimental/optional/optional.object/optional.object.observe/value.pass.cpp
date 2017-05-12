//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: availability=macosx10.12
// XFAIL: availability=macosx10.11
// XFAIL: availability=macosx10.10
// XFAIL: availability=macosx10.9
// XFAIL: availability=macosx10.8
// XFAIL: availability=macosx10.7

// <optional>

// T& optional<T>::value();

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::experimental::optional;
using std::experimental::bad_optional_access;

struct X
{
    X() = default;
    X(const X&) = delete;
    constexpr int test() const {return 3;}
    int test() {return 4;}
};

int main()
{
    {
        optional<X> opt;
        opt.emplace();
        assert(opt.value().test() == 4);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<X> opt;
        try
        {
            opt.value();
            assert(false);
        }
        catch (const bad_optional_access&)
        {
        }
    }
#endif
}
