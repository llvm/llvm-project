//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class... Args> void optional<T>::emplace(Args&&... args);

#include <optional>
#include <type_traits>
#include <cassert>
#include <memory>

#include "test_macros.h"
#include "archetypes.hpp"

using std::optional;

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
    Y(int) { TEST_THROW(6);}
    ~Y() {dtor_called = true;}
};

bool Y::dtor_called = false;

template <class T>
void test_one_arg() {
    using Opt = std::optional<T>;
    {
        Opt opt;
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(0));
    }
    {
        Opt opt;
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1));
    }
    {
        Opt opt(2);
        opt.emplace();
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(0));
    }
    {
        Opt opt(2);
        opt.emplace(1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(1));
    }
}


template <class T>
void test_multi_arg()
{
    test_one_arg<T>();
    using Opt = std::optional<T>;
    Opt opt;
    {
        opt.emplace(101, 41);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(101, 41));
    }
    {
        Opt opt;
        opt.emplace({1, 2, 3, 4});
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(4)); // T sets its value to the size of the init list
    }
    {
        Opt opt;
        opt.emplace({1, 2, 3, 4, 5}, 6);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(5)); // T sets its value to the size of the init list
    }
}

template <class T>
void test_on_test_type() {

    T::reset();
    optional<T> opt;
    assert(T::alive == 0);
    {
        T::reset_constructors();
        opt.emplace();
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::default_constructed == 1);
        assert(T::destroyed == 0);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T());
    }
    {
        T::reset_constructors();
        opt.emplace();
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::default_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T());
    }
    {
        T::reset_constructors();
        opt.emplace(101);
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(101));
    }
    {
        T::reset_constructors();
        opt.emplace(-10, 99);
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(-10, 99));
    }
    {
        T::reset_constructors();
        opt.emplace(-10, 99);
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(-10, 99));
    }
    {
        T::reset_constructors();
        opt.emplace({-10, 99, 42, 1});
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(4)); // size of the initializer list
    }
    {
        T::reset_constructors();
        opt.emplace({-10, 99, 42, 1}, 42);
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(4)); // size of the initializer list
    }
}



int main()
{
    {
        test_on_test_type<TestTypes::TestType>();
        test_on_test_type<ExplicitTestTypes::TestType>();
    }
    {
        using T = int;
        test_one_arg<T>();
        test_one_arg<const T>();
    }
    {
        using T = ConstexprTestTypes::TestType;
        test_multi_arg<T>();
    }
    {
        using T = ExplicitConstexprTestTypes::TestType;
        test_multi_arg<T>();
    }
    {
        using T = TrivialTestTypes::TestType;
        test_multi_arg<T>();
    }
    {
        using T = ExplicitTrivialTestTypes::TestType;
        test_multi_arg<T>();
    }
    {
        optional<const int> opt;
        opt.emplace(42);
        assert(*opt == 42);
        opt.emplace();
        assert(*opt == 0);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    Y::dtor_called = false;
    {
        Y y;
        optional<Y> opt(y);
        try
        {
            assert(static_cast<bool>(opt) == true);
            assert(Y::dtor_called == false);
            opt.emplace(1);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
            assert(Y::dtor_called == true);
        }
    }
#endif
}
