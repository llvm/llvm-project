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

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(F&& f, Args&&... args);

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(launch policy, F&& f, Args&&... args);

#include <future>
#include <memory>
#include <cassert>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds ms;

int f0()
{
    std::this_thread::sleep_for(ms(200));
    return 3;
}

int i = 0;

int& f1()
{
    std::this_thread::sleep_for(ms(200));
    return i;
}

void f2()
{
    std::this_thread::sleep_for(ms(200));
}

std::unique_ptr<int> f3(int i)
{
    std::this_thread::sleep_for(ms(200));
    return std::unique_ptr<int>(new int(i));
}

std::unique_ptr<int> f4(std::unique_ptr<int>&& p)
{
    std::this_thread::sleep_for(ms(200));
    return std::move(p);
}

void f5(int i)
{
    std::this_thread::sleep_for(ms(200));
    throw i;
}

int main()
{
    {
        std::future<int> f = std::async(f0);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int> f = std::async(std::launch::async, f0);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int> f = std::async(std::launch::any, f0);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int> f = std::async(std::launch::deferred, f0);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 > ms(100));
    }

    {
        std::future<int&> f = std::async(f1);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(&f.get() == &i);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int&> f = std::async(std::launch::async, f1);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(&f.get() == &i);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int&> f = std::async(std::launch::any, f1);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(&f.get() == &i);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<int&> f = std::async(std::launch::deferred, f1);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(&f.get() == &i);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 > ms(100));
    }

    {
        std::future<void> f = std::async(f2);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        f.get();
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<void> f = std::async(std::launch::async, f2);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        f.get();
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<void> f = std::async(std::launch::any, f2);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        f.get();
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }
    {
        std::future<void> f = std::async(std::launch::deferred, f2);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        f.get();
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 > ms(100));
    }

    {
        std::future<std::unique_ptr<int>> f = std::async(f3, 3);
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(*f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }

    {
        std::future<std::unique_ptr<int>> f =
                               std::async(f4, std::unique_ptr<int>(new int(3)));
        std::this_thread::sleep_for(ms(300));
        Clock::time_point t0 = Clock::now();
        assert(*f.get() == 3);
        Clock::time_point t1 = Clock::now();
        assert(t1-t0 < ms(100));
    }

    {
        std::future<void> f = std::async(f5, 3);
        std::this_thread::sleep_for(ms(300));
        try { f.get(); assert (false); } catch ( int ex ) {}
    }

    {
        std::future<void> f = std::async(std::launch::deferred, f5, 3);
        std::this_thread::sleep_for(ms(300));
        try { f.get(); assert (false); } catch ( int ex ) {}
    }

}
