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

// optional<T>& operator=(const optional<T>& rhs);

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;

struct AllowConstAssign {
  AllowConstAssign(AllowConstAssign const&) {}
  AllowConstAssign const& operator=(AllowConstAssign const&) const {
      return *this;
  }
};

struct X
{
    static bool throw_now;

    X() = default;
    X(const X&)
    {
        if (throw_now)
            throw 6;
    }
};

bool X::throw_now = false;

int main()
{
    {
        optional<int> opt;
        constexpr optional<int> opt2;
        opt = opt2;
        static_assert(static_cast<bool>(opt2) == false, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<const AllowConstAssign> opt;
        optional<const AllowConstAssign> opt2;
        opt = opt2;
    }
    {
        optional<int> opt;
        constexpr optional<int> opt2(2);
        opt = opt2;
        static_assert(static_cast<bool>(opt2) == true, "");
        static_assert(*opt2 == 2, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }
    {
        optional<int> opt(3);
        constexpr optional<int> opt2;
        opt = opt2;
        static_assert(static_cast<bool>(opt2) == false, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<int> opt(3);
        constexpr optional<int> opt2(2);
        opt = opt2;
        static_assert(static_cast<bool>(opt2) == true, "");
        static_assert(*opt2 == 2, "");
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }
    {
        optional<X> opt;
        optional<X> opt2(X{});
        assert(static_cast<bool>(opt2) == true);
        try
        {
            X::throw_now = true;
            opt = opt2;
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
        }
    }
}
