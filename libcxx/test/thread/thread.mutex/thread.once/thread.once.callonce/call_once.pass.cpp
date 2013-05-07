//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// struct once_flag;

// template<class Callable, class ...Args>
//   void call_once(once_flag& flag, Callable func, Args&&... args);

#include <mutex>
#include <thread>
#include <cassert>

typedef std::chrono::milliseconds ms;

std::once_flag flg0;

int init0_called = 0;

void init0()
{
    std::this_thread::sleep_for(ms(250));
    ++init0_called;
}

void f0()
{
    std::call_once(flg0, init0);
}

std::once_flag flg3;

int init3_called = 0;
int init3_completed = 0;

void init3()
{
    ++init3_called;
    std::this_thread::sleep_for(ms(250));
    if (init3_called == 1)
        throw 1;
    ++init3_completed;
}

void f3()
{
    try
    {
        std::call_once(flg3, init3);
    }
    catch (...)
    {
    }
}

#ifndef _LIBCPP_HAS_NO_VARIADICS

struct init1
{
    static int called;

    void operator()(int i) {called += i;}
};

int init1::called = 0;

std::once_flag flg1;

void f1()
{
    std::call_once(flg1, init1(), 1);
}

struct init2
{
    static int called;

    void operator()(int i, int j) const {called += i + j;}
};

int init2::called = 0;

std::once_flag flg2;

void f2()
{
    std::call_once(flg2, init2(), 2, 3);
    std::call_once(flg2, init2(), 4, 5);
}

#endif  // _LIBCPP_HAS_NO_VARIADICS

std::once_flag flg41;
std::once_flag flg42;

int init41_called = 0;
int init42_called = 0;

void init42();

void init41()
{
    std::this_thread::sleep_for(ms(250));
    ++init41_called;
}

void init42()
{
    std::this_thread::sleep_for(ms(250));
    ++init42_called;
}

void f41()
{
    std::call_once(flg41, init41);
    std::call_once(flg42, init42);
}

void f42()
{
    std::call_once(flg42, init42);
    std::call_once(flg41, init41);
}

#ifndef _LIBCPP_HAS_NO_VARIADICS

class MoveOnly
{
    MoveOnly(const MoveOnly&);
public:
    MoveOnly() {}
    MoveOnly(MoveOnly&&) {}

    void operator()(MoveOnly&&)
    {
    }
};

#endif

int main()
{
    // check basic functionality
    {
        std::thread t0(f0);
        std::thread t1(f0);
        t0.join();
        t1.join();
        assert(init0_called == 1);
    }
    // check basic exception safety
    {
        std::thread t0(f3);
        std::thread t1(f3);
        t0.join();
        t1.join();
        assert(init3_called == 2);
        assert(init3_completed == 1);
    }
    // check deadlock avoidance
    {
        std::thread t0(f41);
        std::thread t1(f42);
        t0.join();
        t1.join();
        assert(init41_called == 1);
        assert(init42_called == 1);
    }
#ifndef _LIBCPP_HAS_NO_VARIADICS
    // check functors with 1 arg
    {
        std::thread t0(f1);
        std::thread t1(f1);
        t0.join();
        t1.join();
        assert(init1::called == 1);
    }
    // check functors with 2 args
    {
        std::thread t0(f2);
        std::thread t1(f2);
        t0.join();
        t1.join();
        assert(init2::called == 5);
    }
    {
        std::once_flag f;
        std::call_once(f, MoveOnly(), MoveOnly());
    }
#endif  // _LIBCPP_HAS_NO_VARIADICS
}
