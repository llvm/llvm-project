//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XFAIL: libcpp-no-exceptions
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class promise<R>

// void promise::set_value(const R& r);

#include <future>
#include <cassert>

struct A
{
    A() {}
    A(const A&) {throw 10;}
};

int main()
{
    {
        typedef int T;
        T i = 3;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(i);
        ++i;
        assert(f.get() == 3);
        --i;
        try
        {
            p.set_value(i);
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
    }
    {
        typedef A T;
        T i;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        try
        {
            p.set_value(i);
            assert(false);
        }
        catch (int j)
        {
            assert(j == 10);
        }
    }
}
