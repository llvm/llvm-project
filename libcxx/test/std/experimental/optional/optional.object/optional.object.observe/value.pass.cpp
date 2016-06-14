//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: libcpp-no-exceptions
// <optional>

// T& optional<T>::value();

#include <experimental/optional>
#include <type_traits>
#include <cassert>

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
}
