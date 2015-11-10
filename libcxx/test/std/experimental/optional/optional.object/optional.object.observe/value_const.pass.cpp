//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <optional>

// constexpr const T& optional<T>::value() const;

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;
using std::experimental::in_place_t;
using std::experimental::in_place;
using std::experimental::bad_optional_access;

struct X
{
    X() = default;
    X(const X&) = delete;
   constexpr int test() const {return 3;}
    int test() {return 4;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        constexpr optional<X> opt(in_place);
        static_assert(opt.value().test() == 3, "");
    }
    {
        const optional<X> opt(in_place);
        assert(opt.value().test() == 3);
    }
    {
        const optional<X> opt;
        try
        {
            opt.value();
            assert(false);
        }
        catch (const bad_optional_access&)
        {
        }
    }
#endif  // _LIBCPP_STD_VER > 11
}
