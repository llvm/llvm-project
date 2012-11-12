//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class promise<R>

// void promise::set_value(R&& r);

#include <future>
#include <memory>
#include <cassert>

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

struct A
{
    A() {}
    A(const A&) = delete;
    A(A&&) {throw 9;}
};

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        typedef std::unique_ptr<int> T;
        T i(new int(3));
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(std::move(i));
        assert(*f.get() == 3);
        try
        {
            p.set_value(std::move(i));
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
            p.set_value(std::move(i));
            assert(false);
        }
        catch (int j)
        {
            assert(j == 9);
        }
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
