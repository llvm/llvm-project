//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// template <class... Args> void optional<T>::emplace(Args&&... args);

#include <experimental/optional>
#include <type_traits>
#include <cassert>
#include <memory>

#include "test_macros.h"

using std::experimental::optional;

class X
{
    int i_;
    int j_ = 0;
public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    friend bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Y
{
public:
    static bool dtor_called;
    Y() = default;
    ~Y() {dtor_called = true;}
};

bool Y::dtor_called = false;

class Z
{
public:
    static bool dtor_called;
    Z() = default;
    Z(int) {TEST_THROW(6);}
    ~Z() {dtor_called = true;}
};

bool Z::dtor_called = false;

int main()
{
    {
        optional<int> opt;
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 0);
    }
    {
        optional<int> opt;
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 1);
    }
    {
        optional<int> opt(2);
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 0);
    }
    {
        optional<int> opt(2);
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 1);
    }
    {
        optional<const int> opt(2);
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == 1);
    }
    {
        optional<X> opt;
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X());
    }
    {
        optional<X> opt;
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(1));
    }
    {
        optional<X> opt;
        opt.emplace(1, 2);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(1, 2));
    }
    {
        optional<X> opt(X{3});
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X());
    }
    {
        optional<X> opt(X{3});
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(1));
    }
    {
        optional<X> opt(X{3});
        opt.emplace(1, 2);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == X(1, 2));
    }
    {
        Y y;
        {
            optional<Y> opt(y);
            assert(Y::dtor_called == false);
            opt.emplace();
            assert(Y::dtor_called == true);
        }
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        Z z;
        optional<Z> opt(z);
        try
        {
            assert(static_cast<bool>(opt) == true);
            assert(Z::dtor_called == false);
            opt.emplace(1);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
            assert(Z::dtor_called == true);
        }
    }
#endif
}
