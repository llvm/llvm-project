//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: libcpp-no-exceptions

// <optional>

// constexpr optional(T&& v);

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;

class X
{
    int i_;
public:
    X(int i) : i_(i) {}
    X(X&& x) : i_(x.i_) {}

    friend bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    constexpr Y(int i) : i_(i) {}
    constexpr Y(Y&& x) : i_(x.i_) {}

    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
};

class Z
{
public:
    Z(int) {}
    Z(Z&&) {throw 6;}
};


int main()
{
    {
        typedef int T;
        constexpr optional<T> opt(T(5));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 5, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef double T;
        constexpr optional<T> opt(T(3));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef X T;
        optional<T> opt(T(3));
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 3);
    }
    {
        typedef Y T;
        constexpr optional<T> opt(T(3));
        static_assert(static_cast<bool>(opt) == true, "");
        static_assert(*opt == 3, "");

        struct test_constexpr_ctor
            : public optional<T>
        {
            constexpr test_constexpr_ctor(T&&) {}
        };
    }
    {
        typedef Z T;
        try
        {
            optional<T> opt(T(3));
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
        }
    }
}
