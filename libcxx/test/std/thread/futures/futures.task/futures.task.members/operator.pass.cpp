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

// class packaged_task<R(ArgTypes...)>

// void operator()(ArgTypes... args);

#include <future>
#include <cassert>

class A
{
    long data_;

public:
    explicit A(long i) : data_(i) {}

    long operator()(long i, long j) const
    {
        if (j == 'z')
            throw A(6);
        return data_ + i + j;
    }
};

void func0(std::packaged_task<double(int, char)> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p(3, 'a');
}

void func1(std::packaged_task<double(int, char)> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p(3, 'z');
}

void func2(std::packaged_task<double(int, char)> p)
{
    p(3, 'a');
    try
    {
        p(3, 'c');
    }
    catch (const std::future_error& e)
    {
        assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
    }
}

void func3(std::packaged_task<double(int, char)> p)
{
    try
    {
        p(3, 'a');
    }
    catch (const std::future_error& e)
    {
        assert(e.code() == make_error_code(std::future_errc::no_state));
    }
}

int main()
{
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        std::thread(func0, std::move(p)).detach();
        assert(f.get() == 105.0);
    }
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        std::thread(func1, std::move(p)).detach();
        try
        {
            f.get();
            assert(false);
        }
        catch (const A& e)
        {
            assert(e(3, 'a') == 106);
        }
    }
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        std::thread t(func2, std::move(p));
        assert(f.get() == 105.0);
        t.join();
    }
    {
        std::packaged_task<double(int, char)> p;
        std::thread t(func3, std::move(p));
        t.join();
    }
}
