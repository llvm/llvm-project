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

// ~promise();

#include <future>
#include <cassert>

int main()
{
    {
        typedef int T;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
            p.set_value(3);
        }
        assert(f.get() == 3);
    }
    {
        typedef int T;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
        }
        try
        {
            T i = f.get();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::broken_promise));
        }
    }

    {
        typedef int& T;
        int i = 4;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
            p.set_value(i);
        }
        assert(&f.get() == &i);
    }
    {
        typedef int& T;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
        }
        try
        {
            T i = f.get();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::broken_promise));
        }
    }

    {
        typedef void T;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
            p.set_value();
        }
        f.get();
        assert(true);
    }
    {
        typedef void T;
        std::future<T> f;
        {
            std::promise<T> p;
            f = p.get_future();
        }
        try
        {
            f.get();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            // LWG 2056 changed the values of future_errc, so if we're using new
            // headers with an old library the error codes won't line up.
            //
            // Note that this particular check only applies to promise<void>
            // since the other specializations happen to be implemented in the
            // header rather than the library.
            assert(
                e.code() == make_error_code(std::future_errc::broken_promise) ||
                e.code() == std::error_code(0, std::future_category()));
        }
    }
}
