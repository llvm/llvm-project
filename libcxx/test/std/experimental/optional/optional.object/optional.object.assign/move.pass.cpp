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

// optional<T>& operator=(optional<T>&& rhs)
//     noexcept(is_nothrow_move_assignable<T>::value &&
//              is_nothrow_move_constructible<T>::value);

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;

struct X
{
    static bool throw_now;

    X() = default;
    X(X&&)
    {
        if (throw_now)
            throw 6;
    }
    X& operator=(X&&) noexcept
    {
        return *this;
    }
};

struct Y {};

bool X::throw_now = false;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        static_assert(std::is_nothrow_move_assignable<optional<int>>::value, "");
        optional<int> opt;
        constexpr optional<int> opt2;
        opt = std::move(opt2);
        static_assert(static_cast<bool>(opt2) == false, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<int> opt;
        constexpr optional<int> opt2(2);
        opt = std::move(opt2);
        static_assert(static_cast<bool>(opt2) == true, "");
        static_assert(*opt2 == 2, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }
    {
        optional<int> opt(3);
        constexpr optional<int> opt2;
        opt = std::move(opt2);
        static_assert(static_cast<bool>(opt2) == false, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<int> opt(3);
        constexpr optional<int> opt2(2);
        opt = std::move(opt2);
        static_assert(static_cast<bool>(opt2) == true, "");
        static_assert(*opt2 == 2, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }
    {
        static_assert(!std::is_nothrow_move_assignable<optional<X>>::value, "");
        optional<X> opt;
        optional<X> opt2(X{});
        assert(static_cast<bool>(opt2) == true);
        try
        {
            X::throw_now = true;
            opt = std::move(opt2);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
        }
    }
    {
        static_assert(std::is_nothrow_move_assignable<optional<Y>>::value, "");
    }
#endif  // _LIBCPP_STD_VER > 11
}
