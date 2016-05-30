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

// optional(optional<T>&& rhs) noexcept(is_nothrow_move_constructible<T>::value);

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;

template <class T>
void
test(optional<T>& rhs, bool is_going_to_throw = false)
{
    static_assert(std::is_nothrow_move_constructible<optional<T>>::value ==
                  std::is_nothrow_move_constructible<T>::value, "");
    bool rhs_engaged = static_cast<bool>(rhs);
    try
    {
        optional<T> lhs = std::move(rhs);
        assert(is_going_to_throw == false);
        assert(static_cast<bool>(lhs) == rhs_engaged);
    }
    catch (int i)
    {
        assert(i == 6);
    }
}

class X
{
    int i_;
public:
    X(int i) : i_(i) {}
    X(X&& x) : i_(x.i_) {x.i_ = 0;}
    ~X() {i_ = 0;}
    friend bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

class Y
{
    int i_;
public:
    Y(int i) : i_(i) {}
    Y(Y&& x) noexcept : i_(x.i_) {x.i_ = 0;}

    friend constexpr bool operator==(const Y& x, const Y& y) {return x.i_ == y.i_;}
};

int count = 0;

class Z
{
    int i_;
public:
    Z(int i) : i_(i) {}
    Z(Z&&)
    {
        if (++count == 2)
            throw 6;
    }

    friend constexpr bool operator==(const Z& x, const Z& y) {return x.i_ == y.i_;}
};

int main()
{
    {
        typedef int T;
        optional<T> rhs;
        test(rhs);
    }
    {
        typedef int T;
        optional<T> rhs(3);
        test(rhs);
    }
    {
        typedef X T;
        optional<T> rhs;
        test(rhs);
    }
    {
        typedef X T;
        optional<T> rhs(X(3));
        test(rhs);
    }
    {
        typedef Y T;
        optional<T> rhs;
        test(rhs);
    }
    {
        typedef Y T;
        optional<T> rhs(Y(3));
        test(rhs);
    }
    {
        typedef Z T;
        optional<T> rhs;
        test(rhs);
    }
    {
        typedef Z T;
        optional<T> rhs(Z(3));
        test(rhs, true);
    }
}
