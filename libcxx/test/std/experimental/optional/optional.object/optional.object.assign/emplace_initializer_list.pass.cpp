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

// template <class U, class... Args>
//   void optional<T>::emplace(initializer_list<U> il, Args&&... args);

#include <experimental/optional>
#include <type_traits>
#include <cassert>
#include <vector>

#include "test_macros.h"

using std::experimental::optional;

class X
{
    int i_;
    int j_ = 0;
public:
    static bool dtor_called;
    constexpr X() : i_(0) {}
    constexpr X(int i) : i_(i) {}
    constexpr X(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}
    ~X() {dtor_called = true;}

    friend constexpr bool operator==(const X& x, const X& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

bool X::dtor_called = false;

class Y
{
    int i_;
    int j_ = 0;
public:
    constexpr Y() : i_(0) {}
    constexpr Y(int i) : i_(i) {}
    constexpr Y(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}

    friend constexpr bool operator==(const Y& x, const Y& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

class Z
{
    int i_;
    int j_ = 0;
public:
    static bool dtor_called;
    constexpr Z() : i_(0) {}
    constexpr Z(int i) : i_(i) {}
    Z(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1])
        {TEST_THROW(6);}
    ~Z() {dtor_called = true;}

    friend constexpr bool operator==(const Z& x, const Z& y)
        {return x.i_ == y.i_ && x.j_ == y.j_;}
};

bool Z::dtor_called = false;

int main()
{
    {
        X x;
        {
            optional<X> opt(x);
            assert(X::dtor_called == false);
            opt.emplace({1, 2});
            assert(X::dtor_called == true);
            assert(*opt == X({1, 2}));
        }
    }
    X::dtor_called = false;
    {
        X x;
        {
            optional<const X> opt(x);
            assert(X::dtor_called == false);
            opt.emplace({1, 2});
            assert(X::dtor_called == true);
            assert(*opt == X({1, 2}));
        }
    }
    {
        optional<std::vector<int>> opt;
        opt.emplace({1, 2, 3}, std::allocator<int>());
        assert(static_cast<bool>(opt) == true);
        assert(*opt == std::vector<int>({1, 2, 3}));
    }
    {
        optional<Y> opt;
        opt.emplace({1, 2});
        assert(static_cast<bool>(opt) == true);
        assert(*opt == Y({1, 2}));
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        Z z;
        optional<Z> opt(z);
        try
        {
            assert(static_cast<bool>(opt) == true);
            assert(Z::dtor_called == false);
            opt.emplace({1, 2});
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
